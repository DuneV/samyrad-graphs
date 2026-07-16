#!/usr/bin/env python3
"""
latent_space.py
===============
Módulo de espacio latente para GNNCostOptimizer.

Tres beneficios sobre el GAT existente:

1. GraphSAGE — embedding de nodo nuevo sin reentrenar
   h_new = σ( W · AGG({h_j : j ∈ N(i)}) )
   donde AGG usa coeficientes de atención α_ij

2. Rayleigh-Ritz incremental — velocidad de inferencia
   En vez de recomputar toda la eigendecomposición O(n³),
   actualiza solo los eigenvectores afectados O(k²)

3. Rebalanceo al fallar — propagación a objetos similares
   Cuando falla grasp(alien), mueve h_alien en E y propaga
   el delta de costo a objetos similares ponderado por α_ij
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lobpcg
from scipy.sparse.csgraph import laplacian
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Optional, Tuple
import networkx as nx
import warnings
warnings.filterwarnings('ignore')


class LatentSpaceManager:
    """
    Gestiona el espacio latente del grafo semántico.
    Se monta encima del GNNCostOptimizer existente sin modificarlo.
    """

    def __init__(self,
                 gnn_optimizer,
                 latent_dim: int = 16,
                 n_eigenvectors: int = 16):
        """
        gnn_optimizer : instancia de GNNCostOptimizer ya inicializada
        latent_dim    : dimensión del espacio latente k
        n_eigenvectors: número de eigenvectores del Laplaciano a mantener
        """
        self.gnn       = gnn_optimizer
        self.k         = latent_dim
        self.n_eig     = n_eigenvectors
        self.encoder   = gnn_optimizer.text_encoder   # SentenceTransformer compartido

        # E ∈ R^{n×k}: matriz de eigenvectores del Laplaciano normalizado
        # Cada fila E[i] es la representación del nodo i en el espacio latente
        self.E: Optional[np.ndarray] = None
        self.node_list: List[str]    = []
        self.node_to_idx: Dict[str, int] = {}

        # Cache de Wh_i = W · h_i para cada nodo conocido
        # W ∈ R^{d'×d} es la matriz de proyección de la primera capa GAT
        self._wh_cache: Dict[str, np.ndarray] = {}

        # Historial de fallos para regularización TWP
        self._failure_history: List[Dict] = []

    # ── 1. Construcción inicial ───────────────────────────────────────────────

    def build(self, semantic_graph: nx.MultiDiGraph) -> None:
        """
        Construye el espacio latente E desde el Laplaciano del grafo.

        Formulación matemática:
        ─────────────────────
        1. Matriz de adyacencia ponderada A ∈ R^{n×n}:
               A_{ij} = 1 / (cost_{ij} + ε)
           Aristas baratas (buenas acciones) → peso alto

        2. Laplaciano normalizado L ∈ R^{n×n}:
               L = D^{-1/2} (D - A) D^{-1/2}
           donde D_{ii} = Σ_j A_{ij}  (grado del nodo i)

        3. Eigendecomposición (LOBPCG = Rayleigh-Ritz):
               L · E = E · Λ
           Los k eigenvectores con eigenvalores más pequeños
           capturan la estructura de comunidades del grafo.
           E ∈ R^{n×k},  Λ = diag(λ_1, ..., λ_k),  λ_1 ≤ λ_2 ≤ ... ≤ λ_k

        Complejidad: O(n³) — solo se ejecuta una vez al inicio.
        """
        self.node_list   = list(semantic_graph.nodes())
        self.node_to_idx = {n: i for i, n in enumerate(self.node_list)}
        n = len(self.node_list)

        print(f"[LatentSpace] Construyendo E: n={n} nodos, k={self.k}")

        # ── A_{ij} = 1 / (cost_{ij} + ε) ─────────────────────────────────────
        A = np.zeros((n, n), dtype=np.float32)
        for u, v, data in semantic_graph.edges(data=True):
            if u in self.node_to_idx and v in self.node_to_idx:
                i, j   = self.node_to_idx[u], self.node_to_idx[v]
                cost   = data.get('adjusted_cost', data.get('cost', 1.0))
                # A_{ij} = 1 / (cost + ε)  →  acciones baratas tienen más peso
                weight = 1.0 / (cost + 1e-6)
                A[i, j] = weight
                A[j, i] = weight    # simétrico para que L sea semidefinido positivo

        # ── L = D^{-1/2} (D - A) D^{-1/2} ────────────────────────────────────
        L_sparse = csr_matrix(laplacian(A, normed=True))

        # ── L·E = E·Λ  via LOBPCG (Locally Optimal Block Preconditioned CG) ──
        # LOBPCG implementa Rayleigh-Ritz:
        #   min_{X} trace(X^T L X) / trace(X^T X)
        # El mínimo se alcanza en los eigenvectores de L con λ más pequeños
        k_actual = min(self.k, n - 1)
        X0       = np.random.randn(n, k_actual).astype(np.float32)

        try:
            eigenvalues, eigenvectors = lobpcg(
                L_sparse, X0,
                largest  = False,   # queremos los k menores eigenvalores
                maxiter  = 300,
                tol      = 1e-6,
            )
            self.E = eigenvectors   # E ∈ R^{n×k}
            print(f"[LatentSpace] E={self.E.shape}, λ=[{eigenvalues[:3].round(4)}...]")
        except Exception as e:
            print(f"[LatentSpace] LOBPCG falló ({e}) → SVD fallback")
            # Fallback: SVD de A como aproximación
            # A ≈ U Σ V^T  →  E = V^T[:k].T
            _, _, Vt = np.linalg.svd(A, full_matrices=False)
            self.E   = Vt[:k_actual].T

        # Pre-computar Wh_i = W · h_i para todos los nodos
        self._precompute_wh(semantic_graph)

    def _precompute_wh(self, semantic_graph: nx.MultiDiGraph) -> None:
        """
        Pre-computa la proyección al espacio latente del GAT para cada nodo.

        Ecuación:
            Wh_i = W · h_i
        donde:
            W ∈ R^{d'×d}  es la matriz de pesos de la primera capa GAT
            h_i ∈ R^d     son las features raw del nodo i
            Wh_i ∈ R^{d'} es la representación en el espacio latente del GAT
        """
        if self.gnn.model is None:
            return

        W = self._get_W()
        if W is None:
            return

        for node in self.node_list:
            h = self._get_raw_features(node, semantic_graph)
            if h is not None:
                # Wh_i = W · h_i  ∈ R^{d'}
                wh = W @ h
                self._wh_cache[node] = wh

        print(f"[LatentSpace] Wh cacheado para {len(self._wh_cache)} nodos")

    def _get_W(self) -> Optional[np.ndarray]:
        """
        Extrae W ∈ R^{d'×d} de la primera capa GATConv.

        En PyTorch Geometric, GATConv tiene:
            lin_src : proyección del nodo fuente  (W_s)
            lin_dst : proyección del nodo destino (W_d)
        Usamos lin_src como W para proyectar todos los nodos al mismo espacio.
        """
        try:
            gat1 = self.gnn.model.gat1
            # W ∈ R^{d'×d}  donde d' = hidden_dim×heads, d = node_feature_dim
            if hasattr(gat1, 'lin_src'):
                return gat1.lin_src.weight.detach().cpu().numpy()
            elif hasattr(gat1, 'lin'):
                return gat1.lin.weight.detach().cpu().numpy()
            else:
                for name, param in self.gnn.model.gat1.named_parameters():
                    if 'weight' in name and param.dim() == 2:
                        return param.detach().cpu().numpy()
        except Exception as e:
            print(f"[LatentSpace] No se pudo extraer W: {e}")
        return None

    def _get_raw_features(self, node: str,
                           semantic_graph: nx.MultiDiGraph) -> Optional[np.ndarray]:
        """
        Extrae h_i ∈ R^d — features raw del nodo i.
        Mismas 9 dimensiones que usa FeatureExtractor:
            h_i = [is_robot, is_safe, is_dangerous, n_affordances,
                   is_detected, confidence, goal_similarity,
                   n_spatial_relations, is_target]
        """
        try:
            node_data     = semantic_graph.nodes[node]
            is_robot      = 1.0 if node_data.get('type') == 'robot_component' else 0.0
            safety        = node_data.get('safety_level', 'safe')
            is_safe       = 1.0 if safety == 'safe'      else 0.0
            is_dangerous  = 1.0 if safety == 'dangerous' else 0.0
            affordances   = node_data.get('affordances', [])
            n_affordances = float(len(affordances))
            is_detected   = float(node_data.get('detected', False))
            confidence    = float(node_data.get('confidence', 0.0))
            ctx           = node_data.get('contextual_info', node)
            emb           = self.encoder.encode(ctx)
            # similitud del concepto con sí mismo = 1 (placeholder para goal_sim)
            sim           = float(np.dot(emb, emb) / (np.linalg.norm(emb)**2 + 1e-8))

            # h_i ∈ R^9
            return np.array([
                is_robot, is_safe, is_dangerous, n_affordances,
                is_detected, confidence, sim, 0.0, 0.0
            ], dtype=np.float32)
        except Exception:
            return None

    # ── 2. GraphSAGE: embedding de nodo nuevo ────────────────────────────────

    def add_node(self,
                 new_concept: str,
                 semantic_graph: nx.MultiDiGraph,
                 known_concepts: Optional[List[str]] = None) -> np.ndarray:
        """
        Calcula el embedding de un nodo nuevo usando GraphSAGE
        con los pesos W del GAT ya entrenado. No reentrena nada.

        Ecuaciones:
        ──────────
        1. Embedding semántico via SentenceTransformer:
               s_new = SentenceTransformer(new_concept) ∈ R^{384}

        2. Similitud coseno con vecinos conocidos:
               sim(new, j) = (s_new · s_j) / (‖s_new‖ · ‖s_j‖)

        3. Coeficientes de atención (softmax sobre similitudes):
               α_{new,j} = exp(sim(new,j)) / Σ_k exp(sim(new,k))
               α ∈ R^{|N|}  con  Σ_j α_{new,j} = 1

        4. Proyección al espacio latente del GAT:
               Wh_j = W · h_j  para cada vecino j conocido

        5. Agregación GraphSAGE con atención:
               Wh_agg = Σ_j α_{new,j} · Wh_j

        6. Combinación con embedding directo del nodo nuevo:
               Wh_new = β · (W · h_new) + (1-β) · Wh_agg
               donde β=0.4 balancea conocimiento directo vs heredado

        7. Actualización incremental de E via Rayleigh-Ritz (paso 3)

        Complejidad: O(|N| · d') donde |N| = vecinos, d' = dim latente GAT
        """
        if self.E is None:
            self.build(semantic_graph)

        W = self._get_W()
        if W is None or not self._wh_cache:
            print(f"[LatentSpace] W no disponible → embedding aleatorio")
            return np.random.randn(self.k).astype(np.float32) * 0.1

        # ── s_new = SentenceTransformer(new_concept) ───────────────────────────
        emb_new   = self.encoder.encode(new_concept)
        neighbors = known_concepts or list(self._wh_cache.keys())
        neighbors = [n for n in neighbors if n in self._wh_cache]

        if not neighbors:
            return np.zeros(self.k, dtype=np.float32)

        # ── sim(new, j) = (s_new · s_j) / (‖s_new‖ · ‖s_j‖) ─────────────────
        similarities = np.array([
            float(np.dot(emb_new, self.encoder.encode(n)) / (
                np.linalg.norm(emb_new) * np.linalg.norm(self.encoder.encode(n)) + 1e-8
            ))
            for n in neighbors
        ], dtype=np.float32)

        # ── α_{new,j} = exp(sim_j) / Σ_k exp(sim_k)  (softmax) ───────────────
        alphas = np.exp(similarities) / np.sum(np.exp(similarities))
        # α ∈ R^{|N|},  Σ_j α_j = 1

        # ── Wh_j = W · h_j  para vecinos conocidos ────────────────────────────
        Wh_neighbors = np.stack([self._wh_cache[n] for n in neighbors])  # (|N|, d')

        # ── Wh_agg = Σ_j α_j · Wh_j  (agregación ponderada) ──────────────────
        Wh_agg = alphas @ Wh_neighbors   # (d',)  — producto α^T · Wh

        # ── Wh_new = β·(W·h_new) + (1-β)·Wh_agg ──────────────────────────────
        h_raw_new = self._get_raw_features(new_concept, semantic_graph)
        beta      = 0.4   # β: balance conocimiento directo vs heredado de vecinos

        if h_raw_new is not None:
            Wh_direct = W @ h_raw_new   # W·h_new ∈ R^{d'}
            d         = min(len(Wh_direct), len(Wh_agg))
            Wh_final  = beta * Wh_direct[:d] + (1 - beta) * Wh_agg[:d]
        else:
            Wh_final = Wh_agg

        # ── Actualizar E con el nuevo nodo via Rayleigh-Ritz ───────────────────
        h_latent = self._rayleigh_ritz_update(new_concept, Wh_final, semantic_graph)

        # Guardar en cache
        self._wh_cache[new_concept]      = Wh_final
        self.node_to_idx[new_concept]    = len(self.node_list)
        self.node_list.append(new_concept)

        # Log top-3 vecinos más influyentes
        top3 = sorted(zip(neighbors, alphas), key=lambda x: x[1], reverse=True)[:3]
        top3_str = [(n, round(float(a), 3)) for n, a in top3]
        print(f"[LatentSpace] '{new_concept}' → α_top3={top3_str}")

        return h_latent

    # ── 3. Rayleigh-Ritz incremental ─────────────────────────────────────────

    def _rayleigh_ritz_update(self,
                               new_concept: str,
                               wh_new: np.ndarray,
                               semantic_graph: nx.MultiDiGraph) -> np.ndarray:
        """
        Actualiza E con el nuevo nodo sin recomputar toda la eigendecomposición.

        Ecuaciones (actualización de rango 1):
        ──────────────────────────────────────
        1. Vector de conexiones c ∈ R^n:
               c_j = sim(Wh_new, Wh_j) = (Wh_new · Wh_j) / (‖Wh_new‖·‖Wh_j‖)
               c_j > 0 solo si la similitud es positiva

        2. Coordenadas del nuevo nodo en el subespacio actual E:
               h_latent = E^T · c  ∈ R^k
               (proyección de c sobre los k eigenvectores actuales)

        3. Normalización:
               h_latent ← h_latent / ‖h_latent‖

        4. Actualización de rango 1 de E (Rayleigh-Ritz local):
               ΔE = (c / ‖c‖) · h_latent^T  ∈ R^{n×k}
               E_new = E_old + η · ΔE
           Solo las filas de E conectadas al nuevo nodo cambian.

        5. Re-ortogonalización QR (mantiene E ortogonal):
               E_new, _ = QR(E_new)
               Garantiza E^T·E = I  —  O(k²) en vez de O(n³)

        Complejidad total: O(n·k + k²) ≪ O(n³) de eigendecomposición completa
        """
        if self.E is None:
            # Sin E, devolver proyección directa de Wh al espacio de dim k
            d = len(wh_new)
            if d >= self.k:
                return wh_new[:self.k]
            return np.pad(wh_new, (0, self.k - d))

        n_old = self.E.shape[0]
        k     = self.E.shape[1]

        # ── c_j = sim(Wh_new, Wh_j) para j ∈ nodos existentes ─────────────────
        connections = np.zeros(n_old, dtype=np.float32)
        for neighbor, wh in self._wh_cache.items():
            if neighbor == new_concept or neighbor not in self.node_to_idx:
                continue
            idx = self.node_to_idx[neighbor]
            if idx < n_old:
                # c_j = (Wh_new · Wh_j) / (‖Wh_new‖ · ‖Wh_j‖)
                sim = float(np.dot(wh_new, wh) / (
                    np.linalg.norm(wh_new) * np.linalg.norm(wh) + 1e-8
                ))
                connections[idx] = max(0.0, sim)   # c_j ≥ 0

        # ── h_latent = E^T · c  (proyección sobre subespacio actual) ───────────
        # h_latent ∈ R^k: coordenadas del nuevo nodo en el espacio latente E
        h_latent = self.E.T @ connections   # R^{k×n} · R^n = R^k

        # Normalizar: h_latent ← h_latent / ‖h_latent‖
        norm = np.linalg.norm(h_latent)
        if norm > 1e-6:
            h_latent /= norm

        # ── ΔE = (c/‖c‖) · h_latent^T  (actualización de rango 1) ────────────
        c_norm  = connections / (np.linalg.norm(connections) + 1e-6)
        delta_E = np.outer(c_norm, h_latent)   # R^{n×k}

        # E_new = E_old + η · ΔE
        eta_E    = 0.05   # learning rate pequeño para no distorsionar E
        self.E   = self.E + eta_E * delta_E[:self.E.shape[0], :self.E.shape[1]]

        # ── QR para mantener E^T·E = I  —  O(k²) ──────────────────────────────
        # QR descompone E = Q·R donde Q tiene columnas ortonormales
        # Reemplazamos E con Q para mantener la propiedad de eigenvectores
        self.E, _ = np.linalg.qr(self.E)

        return h_latent

    # ── 4. Rebalanceo al fallar ───────────────────────────────────────────────

    def rebalance_on_failure(self,
                              failed_node: str,
                              failed_action: str,
                              semantic_graph: nx.MultiDiGraph,
                              eta: float = 0.05) -> Dict[str, float]:
        """
        Cuando falla una acción, rebalancea en el espacio latente E
        y propaga el delta de costo a nodos similares.

        Ecuaciones:
        ──────────
        1. Dirección del fallo en E — hacia zona de nodos peligrosos:
               d = centroide({E[j] : safety(j)='dangerous'}) - E[i]
               d ← d / ‖d‖

        2. Mover el nodo fallido en E:
               E[i] ← E[i] + η · d
           El nodo se aleja de la zona "safe" hacia la zona "dangerous"

        3. Similitud en espacio Wh entre nodo fallido y vecinos:
               sim(i, j) = (Wh_i · Wh_j) / (‖Wh_i‖ · ‖Wh_j‖)

        4. Propagación proporcional a la similitud:
               Δcost_j = sim(i,j) · 2.0  si sim(i,j) > 0.7
           Nodos muy similares al que falló también reciben penalización.
           Nodos poco similares no se ven afectados.

        5. Mover vecinos similares en E (propagación suave):
               E[j] ← E[j] + η · sim(i,j) · 0.3 · d
           Factor 0.3: propagación más débil que el nodo original

        Retorna: {nodo: Δcost} para actualizar adjusted_cost en el grafo
        """
        if self.E is None or failed_node not in self.node_to_idx:
            return {}

        failed_idx = self.node_to_idx[failed_node]

        # ── d = centroide(dangerous) - E[i]  ──────────────────────────────────
        dangerous_nodes = [
            n for n in self.node_list
            if semantic_graph.has_node(n) and
               semantic_graph.nodes[n].get('safety_level') == 'dangerous'
        ]

        direction = np.zeros(self.k, dtype=np.float32)

        if dangerous_nodes:
            dangerous_indices = [
                self.node_to_idx[n] for n in dangerous_nodes
                if n in self.node_to_idx and self.node_to_idx[n] < len(self.E)
            ]
            if dangerous_indices:
                # centroide = (1/|D|) · Σ_{j∈D} E[j]
                danger_centroid  = np.mean(self.E[dangerous_indices], axis=0)
                # d = centroide - E[i]
                direction        = danger_centroid - self.E[failed_idx]
                # d ← d / ‖d‖
                norm_d           = np.linalg.norm(direction)
                if norm_d > 1e-6:
                    direction /= norm_d

                # ── E[i] ← E[i] + η · d  ──────────────────────────────────────
                self.E[failed_idx] += eta * direction

        # Guardar en historial
        self._failure_history.append({
            'node': failed_node, 'action': failed_action, 'idx': failed_idx
        })

        # ── Δcost_j = sim(i,j) · 2.0  si sim(i,j) > 0.7 ──────────────────────
        delta_costs: Dict[str, float] = {}
        wh_failed   = self._wh_cache.get(failed_node)

        if wh_failed is not None:
            for node, wh in self._wh_cache.items():
                if node == failed_node:
                    # El nodo fallido recibe penalización completa
                    delta_costs[node] = 5.0
                    continue

                # sim(i,j) = (Wh_i · Wh_j) / (‖Wh_i‖ · ‖Wh_j‖)
                sim = float(np.dot(wh_failed, wh) / (
                    np.linalg.norm(wh_failed) * np.linalg.norm(wh) + 1e-8
                ))

                if sim > 0.7:
                    # Δcost_j = sim · 2.0  (propagación proporcional)
                    delta_costs[node] = sim * 2.0

                    # ── E[j] ← E[j] + η · sim · 0.3 · d  ─────────────────────
                    if node in self.node_to_idx and np.any(direction != 0):
                        idx = self.node_to_idx[node]
                        if idx < len(self.E):
                            self.E[idx] += eta * sim * 0.3 * direction

        print(f"[LatentSpace] Fallo '{failed_action}({failed_node})' → "
              f"{len(delta_costs)} nodos afectados")
        top3 = sorted(delta_costs.items(), key=lambda x: x[1], reverse=True)[:3]
        for node, delta in top3:
            print(f"  Δcost({node}) = +{delta:.2f}")

        return delta_costs

    # ── 5. Interpolación de goal nuevo ───────────────────────────────────────

    def interpolate_goal(self,
                          new_goal: str,
                          edge: Tuple[str, str, str],
                          semantic_graph: nx.MultiDiGraph) -> float:
        """
        Para un goal nunca visto, interpola el factor de coste de una arista
        desde goals similares conocidos en el espacio latente.

        Ecuaciones:
        ──────────
        1. Embedding del goal nuevo:
               s_goal = SentenceTransformer(new_goal) ∈ R^{384}

        2. Similitud coseno con goals conocidos {g_1, ..., g_m}:
               sim(new, g_k) = (s_goal · s_{g_k}) / (‖s_goal‖ · ‖s_{g_k}‖)

        3. Pesos de interpolación (softmax):
               α_k = exp(sim(new, g_k)) / Σ_j exp(sim(new, g_j))

        4. Factor interpolado para la arista (source, action, target):
               factor = Σ_k α_k · factor_k
           donde factor_k es el factor conocido del goal g_k sobre esa arista.

        Ejemplo:
               new_goal = "ferment the dough"
               sim con "mix the dough" = 0.71  → α = 0.52
               sim con "pour the water" = 0.68 → α = 0.48
               factor = 0.52·0.2 + 0.48·1.5 = 0.824
        """
        source, action, target = edge

        # ── s_goal = SentenceTransformer(new_goal) ─────────────────────────────
        emb_new = self.encoder.encode(new_goal)

        # Goals de referencia: nombres de nodos como proxy
        known_goals = [
            f"Move the {n} to the table"
            for n in self.node_list if not n.startswith('robot_')
        ][:20]

        if not known_goals:
            return 1.0   # factor neutro

        # ── sim(new, g_k) = (s_goal · s_{g_k}) / (‖s_goal‖ · ‖s_{g_k}‖) ──────
        similarities = np.array([
            float(np.dot(emb_new, self.encoder.encode(kg)) / (
                np.linalg.norm(emb_new) *
                np.linalg.norm(self.encoder.encode(kg)) + 1e-8
            ))
            for kg in known_goals
        ], dtype=np.float32)

        # ── α_k = exp(sim_k) / Σ_j exp(sim_j)  (softmax) ─────────────────────
        alphas = np.exp(similarities) / np.sum(np.exp(similarities))

        # ── factor_k heurístico: bajo si el nodo aparece en el goal ────────────
        factors = np.array([
            0.2 if any(
                n in kg for n in [source, target]
                if not n.startswith('robot_')
            ) else 1.5
            for kg in known_goals
        ], dtype=np.float32)

        # ── factor = Σ_k α_k · factor_k  (interpolación ponderada) ────────────
        factor_interp = float(np.dot(alphas, factors))
        factor_interp = float(np.clip(factor_interp, 0.1, 2.0))

        print(f"[LatentSpace] '{new_goal}' → "
              f"factor({source},{action},{target})={factor_interp:.3f}")

        return factor_interp

    # ── utilidades ────────────────────────────────────────────────────────────

    def get_similar_nodes(self, concept: str,
                           top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Retorna los top_k nodos más similares en el espacio Wh.

        Métrica:
            sim(i, j) = (Wh_i · Wh_j) / (‖Wh_i‖ · ‖Wh_j‖)
        """
        if concept not in self._wh_cache:
            return []

        wh_query = self._wh_cache[concept]
        scores   = [
            (node, float(np.dot(wh_query, wh) / (
                np.linalg.norm(wh_query) * np.linalg.norm(wh) + 1e-8
            )))
            for node, wh in self._wh_cache.items()
            if node != concept
        ]
        return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]

    def summary(self) -> Dict:
        """Resumen del estado del espacio latente."""
        return {
            'n_nodes':         len(self.node_list),
            'latent_dim_k':    self.k,
            'E_shape':         list(self.E.shape) if self.E is not None else None,
            'wh_cached':       len(self._wh_cache),
            'failure_history': len(self._failure_history),
        }