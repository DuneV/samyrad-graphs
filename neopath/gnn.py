from typing import Dict, Tuple, List
import numpy as np
import networkx as nx
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from sentence_transformers import SentenceTransformer
from neopath.semantic_knowledge import KnowledgeBase
from neopath.perceptual_knowledge import PerceptualGraph
from neopath.semantic_graph import SemanticActionGraph
from sklearn.preprocessing import StandardScaler

class FeatureExtractor:
    """Extrae features contextuales de nodos y aristas para la GNN"""
    
    def __init__(self, text_encoder):
        self.text_encoder = text_encoder
        self.embedding_cache = {}
        self.node_scaler = StandardScaler()
        self.edge_scaler = StandardScaler()
        self.fitted = False
    
    def encode_cached(self, text):
        """Devuelve embedding de texto, usando cache si ya existe"""
        if text not in self.embedding_cache:
            self.embedding_cache[text] = self.text_encoder.encode(text)
        return self.embedding_cache[text]
    
    def fit_scalers(self, semantic_graph, perceptual_graph, goals_sample):
        """Ajusta los normalizadores con múltiples ejemplos"""
        print("Ajustando normalizadores de features...")
        
        node_features_list = []
        edge_features_list = []
        
        for goal in goals_sample:
            node_feat = self._extract_node_features_raw(
                semantic_graph, perceptual_graph, goal, target_objects=None
            )
            edge_feat = self._extract_edge_features_raw(
                semantic_graph, perceptual_graph, goal, 
                target_objects=None, required_actions=None
            )
            
            node_features_list.append(node_feat)
            edge_features_list.append(edge_feat)
        
        all_node_features = np.vstack(node_features_list)
        all_edge_features = np.vstack(edge_features_list)
        
        self.node_scaler.fit(all_node_features)
        self.edge_scaler.fit(all_edge_features)
        self.fitted = True
        
        print(f"✓ Normalizadores ajustados con {len(goals_sample)} ejemplos")
        
    def _extract_node_features_raw(self, semantic_graph: nx.MultiDiGraph, 
                            perceptual_graph: PerceptualGraph,
                            goal: str,
                            target_objects: List[str] = None) -> np.ndarray:
        """Extrae features de nodos sin normalización"""
        goal_embedding = self.encode_cached(goal)
        detected_concepts = set(inst.concept for inst in perceptual_graph.instances.values())
        target_set = set(target_objects) if target_objects else set()
        
        node_features = []
        for node_id in semantic_graph.nodes():
            node_data = semantic_graph.nodes[node_id]
            
            is_robot = 1.0 if node_data.get('type') == 'robot_component' else 0.0
            safety = node_data.get('safety_level', 'safe')
            is_safe = 1.0 if safety == 'safe' else 0.0
            is_dangerous = 1.0 if safety == 'dangerous' else 0.0
            affordances = node_data.get('affordances', [])
            n_affordances = float(len(affordances))
            
            concept_name = node_id.split('_')[0] if '_' in node_id else node_id
            is_detected = 1.0 if concept_name in detected_concepts else 0.0
            
            detection_confidence = 0.0
            matching_instances = [
                inst for inst in perceptual_graph.instances.values() 
                if inst.concept == concept_name
            ]
            if matching_instances:
                detection_confidence = np.mean([inst.confidence for inst in matching_instances])
            
            context_text = node_data.get('contextual_info', node_id)
            context_embedding = self.encode_cached(context_text)
            goal_similarity = float(np.dot(context_embedding, goal_embedding) / (
                np.linalg.norm(context_embedding) * np.linalg.norm(goal_embedding) + 1e-8
            ))
            
            n_spatial_relations = 0.0
            if matching_instances:
                n_spatial_relations = float(np.mean([
                    sum(len(rels) for rels in inst.relations.values())
                    for inst in matching_instances
                ]))
            
            is_target = 1.0 if concept_name in target_set else 0.0
            
            features = [
                is_robot, is_safe, is_dangerous, n_affordances,
                is_detected, detection_confidence, goal_similarity,
                n_spatial_relations, is_target
            ]
            
            node_features.append(features)
        
        return np.array(node_features, dtype=np.float32)
    
    def extract_node_features(self, semantic_graph: nx.MultiDiGraph, 
                        perceptual_graph: PerceptualGraph,
                        goal: str,
                        target_objects: List[str] = None) -> np.ndarray:
        """Extrae features de nodos CON normalización"""
        features = self._extract_node_features_raw(
            semantic_graph, perceptual_graph, goal, target_objects
        )
        
        if self.fitted:
            features = self.node_scaler.transform(features)
        
        return features.astype(np.float32)
    
    def _extract_edge_features_raw(self, semantic_graph: nx.MultiDiGraph,
                            perceptual_graph: PerceptualGraph,
                            goal: str,
                            target_objects: List[str] = None,
                            required_actions: List[str] = None) -> np.ndarray:
        """Extrae features de aristas sin normalización"""
        goal_embedding = self.encode_cached(goal)
        detected_concepts = set(inst.concept for inst in perceptual_graph.instances.values())
        target_set = set(target_objects) if target_objects else set()
        required_action_set = set(required_actions) if required_actions else set()
        
        concept_instances = {}
        for inst in perceptual_graph.instances.values():
            if inst.concept not in concept_instances:
                concept_instances[inst.concept] = []
            concept_instances[inst.concept].append(inst)
        
        edge_features = []
        for u, v, key, data in semantic_graph.edges(keys=True, data=True):
            base_cost = float(data.get('cost', 1.0))
            ethical_weight = float(data.get('ethical_weight', 1.0))
            
            action = data.get('action', 'unknown')
            action_embedding = self.encode_cached(action)
            action_goal_similarity = float(np.dot(action_embedding, goal_embedding) / (
                np.linalg.norm(action_embedding) * np.linalg.norm(goal_embedding) + 1e-8
            ))
            
            source_concept = u.split('_')[0] if '_' in u else u
            target_concept = v.split('_')[0] if '_' in v else v
            
            source_detected = 1.0 if source_concept in detected_concepts else 0.0
            target_detected = 1.0 if target_concept in detected_concepts else 0.0
            
            spatial_feasibility = 0.5
            if source_concept in concept_instances and target_concept in concept_instances:
                distances = []
                for src_inst in concept_instances[source_concept]:
                    for tgt_inst in concept_instances[target_concept]:
                        dist = src_inst.distance_to(tgt_inst)
                        if dist > 0:
                            distances.append(dist)
                
                if distances:
                    avg_distance = np.mean(distances)
                    spatial_feasibility = 1.0 / (1.0 + avg_distance)
            
            spatial_relations_count = 0.0
            if source_concept in concept_instances and target_concept in concept_instances:
                for src_inst in concept_instances[source_concept]:
                    for tgt_inst in concept_instances[target_concept]:
                        for rel_type, targets in src_inst.relations.items():
                            if tgt_inst.id in targets:
                                spatial_relations_count += 1.0
            
            connects_targets = 1.0 if (target_concept in target_set) else 0.0
            is_required_action = 1.0 if action in required_action_set else 0.0
            
            features = [
                base_cost, ethical_weight, action_goal_similarity,
                source_detected, target_detected, spatial_feasibility,
                spatial_relations_count, connects_targets, is_required_action
            ]
            
            edge_features.append(features)
        
        return np.array(edge_features, dtype=np.float32)

    def extract_edge_features(self, semantic_graph: nx.MultiDiGraph,
                        perceptual_graph: PerceptualGraph,
                        goal: str,
                        target_objects: List[str] = None,
                        required_actions: List[str] = None) -> np.ndarray:
        """Extrae features de aristas CON normalización"""
        features = self._extract_edge_features_raw(
            semantic_graph, perceptual_graph, goal, 
            target_objects, required_actions
        )
        
        if self.fitted:
            features = self.edge_scaler.transform(features)
        
        return features.astype(np.float32)


class ContextualEdgeWeightGNN(nn.Module):
    """GNN que ajusta costos de aristas según contexto perceptual y objetivo"""
    
    def __init__(self, node_feature_dim, edge_feature_dim, 
                 goal_embedding_dim=384, hidden_dim=128):
        super(ContextualEdgeWeightGNN, self).__init__()
        
        self.gat1 = GATConv(node_feature_dim, hidden_dim, heads=3, concat=True, dropout=0.1)
        self.gat2 = GATConv(hidden_dim * 3, hidden_dim, heads=2, concat=False, dropout=0.1)
        
        self.bn1 = nn.BatchNorm1d(hidden_dim * 3)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        self.goal_processor = nn.Sequential(
            nn.Linear(goal_embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.edge_predictor = nn.Sequential(
            nn.Linear(edge_feature_dim + 2 * hidden_dim + hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, data, goal_embedding):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        x = self.gat1(x, edge_index)
        if x.size(0) > 1:
            x = self.bn1(x)
        x = F.elu(x)
        x = F.dropout(x, p=0.1, training=self.training)

        x = self.gat2(x, edge_index)
        if x.size(0) > 1:
            x = self.bn2(x)
        x = F.elu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        
        goal_context = self.goal_processor(goal_embedding)
        goal_context = goal_context.unsqueeze(0).expand(edge_index.size(1), -1)
        
        row, col = edge_index
        edge_embeddings = torch.cat([
            edge_attr,
            x[row],
            x[col],
            goal_context
        ], dim=1)

        log_factors = self.edge_predictor(edge_embeddings).squeeze()
        
        factors = 0.1 + F.softplus(log_factors) * 0.5
        factors = torch.clamp(factors, min=0.1, max=2.0)
        
        return factors


class WeightedHuberLoss(nn.Module):
    """Huber loss con pesos para penalizar errores en factores pequeños"""
    
    def __init__(self, delta=0.15):
        super(WeightedHuberLoss, self).__init__()
        self.delta = delta
    
    def forward(self, pred, target):
        weights = 1.0 / (target + 0.1)
        weights = weights / weights.mean()
        
        diff = torch.abs(pred - target)
        loss = torch.where(
            diff < self.delta,
            0.5 * diff ** 2,
            self.delta * (diff - 0.5 * self.delta)
        )
        
        return (loss * weights).mean()


class GNNCostOptimizer:
    """
    Sistema que integra GNN con el grafo semántico para optimizar costos.
    SOLO INFERENCIA - No entrena durante el uso.
    """
    
    def __init__(self, knowledge_base: KnowledgeBase, pretrained_path: str = None):
        self.knowledge_base = knowledge_base
        self.text_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.feature_extractor = FeatureExtractor(self.text_encoder)
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.embedding_cache = {}
        self.is_trained = True
        
        # Cargar modelo pre-entrenado si existe
        if pretrained_path and os.path.exists(pretrained_path):
            # Nota: el modelo debe ser inicializado primero antes de cargar pesos
            print(f"Modelo pre-entrenado encontrado en {pretrained_path}")
            print("Se cargará después de la primera inicialización.")
            self._pretrained_path = pretrained_path
        else:
            self._pretrained_path = None

    def encode_cached(self, text):
        """Devuelve embedding de texto, usando cache si ya existe"""
        if text not in self.embedding_cache:
            self.embedding_cache[text] = self.text_encoder.encode(text)
        return self.embedding_cache[text]
    
    def prepare_graph_data(self, semantic_graph: nx.MultiDiGraph,
                          perceptual_graph: PerceptualGraph,
                          goal: str,
                          target_objects: List[str] = None,
                          required_actions: List[str] = None) -> Tuple[Data, torch.Tensor, Dict]:
        """Convierte grafos a formato PyTorch Geometric"""
        
        node_features = self.feature_extractor.extract_node_features(
            semantic_graph, perceptual_graph, goal, target_objects
        )
        
        edge_features = self.feature_extractor.extract_edge_features(
            semantic_graph, perceptual_graph, goal, target_objects, required_actions
        )
        
        node_list = list(semantic_graph.nodes())
        node_to_idx = {node: i for i, node in enumerate(node_list)}
        
        edge_list = []
        for u, v, key in semantic_graph.edges(keys=True):
            edge_list.append([node_to_idx[u], node_to_idx[v]])
        
        x = torch.tensor(node_features, dtype=torch.float).to(self.device)
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous().to(self.device)
        edge_attr = torch.tensor(edge_features, dtype=torch.float).to(self.device)
        
        goal_embedding = torch.tensor(
            self.encode_cached(goal),
            dtype=torch.float
        ).to(self.device)
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        
        return data, goal_embedding, node_to_idx
    
    def initialize_model(self, semantic_graph: nx.MultiDiGraph,
                        perceptual_graph: PerceptualGraph,
                        goal: str):
        """Inicializa modelo con dimensiones correctas"""
        data, goal_embedding, _ = self.prepare_graph_data(
            semantic_graph, perceptual_graph, goal
        )
        
        self.model = ContextualEdgeWeightGNN(
            node_feature_dim=data.x.shape[1],
            edge_feature_dim=data.edge_attr.shape[1],
            goal_embedding_dim=goal_embedding.shape[0],
            hidden_dim=128
        ).to(self.device)
        
        print(f"✓ GNN model initialized on {self.device}")
        
        # Cargar pesos pre-entrenados si existen
        if self._pretrained_path:
            self.load_model(self._pretrained_path)
    
    def optimize_costs(self, semantic_graph: nx.MultiDiGraph,
                      perceptual_graph: PerceptualGraph,
                      goal: str,
                      target_objects: List[str] = None,
                      required_actions: List[str] = None) -> nx.MultiDiGraph:
        """
        INFERENCIA ÚNICAMENTE - Actualiza costos del grafo semántico usando GNN.
        """
        if not self.is_trained:
            raise RuntimeError(
                "El modelo GNN no está entrenado."
                "Ejecuta GNNTrainer.train() primero o carga un modelo pre-entrenado."
            )
        
        if self.model is None:
            self.initialize_model(semantic_graph, perceptual_graph, goal)
        
        self.model.eval()
        
        data, goal_embedding, node_to_idx = self.prepare_graph_data(
            semantic_graph, perceptual_graph, goal, target_objects, required_actions
        )
        
        with torch.no_grad():
            weight_factors = self.model(data, goal_embedding).cpu().numpy()
        
        G_optimized = semantic_graph.copy()
        
        for i, (u, v, key, edge_data) in enumerate(G_optimized.edges(keys=True, data=True)):
            original_cost = edge_data['cost']
            adjusted_cost = original_cost * weight_factors[i]
            
            G_optimized[u][v][key]['adjusted_cost'] = float(adjusted_cost)
            G_optimized[u][v][key]['cost_factor'] = float(weight_factors[i])
        
        return G_optimized
    
    def save_model(self, path: str = "gnn_cost_optimizer.pth"):
        """Guarda el modelo entrenado"""
        if self.model is not None:
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'is_trained': self.is_trained
            }
            torch.save(checkpoint, path)
            print(f"✓ Model saved to {path}")
        else:
            print("No hay modelo para guardar")
    
    def load_model(self, path: str = "gnn_cost_optimizer.pth"):
        """Carga un modelo pre-entrenado"""
        if self.model is not None and os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.is_trained = checkpoint.get('is_trained', True)
            self.model.eval()
            print(f"✓ Modelo GNN cargado desde {path}")
        elif not os.path.exists(path):
            print(f"No se encontró el archivo {path}")
        else:
            print("Modelo no inicializado. Llama a initialize_model() primero.")


class GNNTrainer:
    """Sistema para entrenar el GNN con datos sintéticos o supervisados"""
    
    def __init__(self, gnn_optimizer: GNNCostOptimizer, 
                 semantic_graph: SemanticActionGraph,
                 knowledge_base: KnowledgeBase):
        self.gnn_optimizer = gnn_optimizer
        self.semantic_graph = semantic_graph
        self.knowledge_base = knowledge_base
    
    def train_supervised(self, labeled_examples: List[Dict], 
                        epochs: int = 200,
                        validation_split: float = 0.2,
                        early_stopping_patience: int = 20,
                        save_path: str = "gnn_cost_optimizer.pth"):
        """
        Entrena con ejemplos etiquetados explícitamente.
        
        Formato de labeled_examples:
        [
            {
                'goal': 'Move cup to table',
                'target_objects': ['cup', 'table'],
                'required_actions': ['grasp', 'move'],
                'detected_objects': ['cup', 'table', 'chair'],
                'object_positions': {
                    'cup': (200, 250, 1.1),
                    'table': (350, 300, 1.4),
                    ...
                },
                'edge_labels': {
                    ('robot_gripper', 'grasp', 'cup'): 0.25,
                    ('cup', 'move', 'table'): 0.35,
                    ...
                }
            },
            ...
        ]
        """
        print(f"\n{'='*60}")
        print(f"ENTRENAMIENTO SUPERVISADO GNN")
        print(f"{'='*60}")
        
        training_data = []
        for example in labeled_examples:
            training_example = self._convert_labeled_to_training(example)
            if training_example:
                training_data.append(training_example)
        
        if not training_data:
            print("❌ Error: No hay datos de entrenamiento válidos")
            return
        
        # Dividir train/val
        n_val = int(len(training_data) * validation_split)
        np.random.shuffle(training_data)
        val_data = training_data[:n_val]
        train_data = training_data[n_val:]
        
        print(f"Train: {len(train_data)} | Val: {len(val_data)}\n")
        
        # Ajustar normalizadores
        sample_goals = [ex['goal'] for ex in training_data[:min(10, len(training_data))]]
        self.gnn_optimizer.feature_extractor.fit_scalers(
            self.semantic_graph.graph,
            train_data[0]['perceptual_graph'],
            sample_goals
        )
        
        # Inicializar modelo
        if self.gnn_optimizer.model is None:
            first_example = training_data[0]
            self.gnn_optimizer.initialize_model(
                first_example['semantic_graph'],
                first_example['perceptual_graph'],
                first_example['goal']
            )
        
        # Configurar entrenamiento
        optimizer = torch.optim.AdamW(
            self.gnn_optimizer.model.parameters(), 
            lr=0.001,
            weight_decay=0.01
        )
        criterion = WeightedHuberLoss(delta=0.15)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Loop de entrenamiento
        for epoch in range(epochs):
            self.gnn_optimizer.model.train()
            total_loss = 0
            batch_count = 0
            
            np.random.shuffle(train_data)
            
            for example in train_data:
                optimizer.zero_grad()
                
                try:
                    data, goal_emb, _ = self.gnn_optimizer.prepare_graph_data(
                        example['semantic_graph'],
                        example['perceptual_graph'],
                        example['goal']
                    )
                    
                    target = torch.tensor(
                        example['optimal_factors'],
                        dtype=torch.float
                    ).to(self.gnn_optimizer.device)
                    
                    pred_factors = self.gnn_optimizer.model(data, goal_emb)
                    loss = criterion(pred_factors, target)
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.gnn_optimizer.model.parameters(), 
                        max_norm=1.0
                    )
                    optimizer.step()
                    
                    total_loss += loss.item()
                    batch_count += 1
                    
                except Exception as e:
                    print(f"❌ Error en ejemplo: {e}")
                    continue
            
            avg_loss = total_loss / batch_count if batch_count > 0 else float('inf')
            
            # Validación
            self.gnn_optimizer.model.eval()
            val_loss = 0
            val_count = 0
            
            with torch.no_grad():
                for example in val_data:
                    try:
                        data, goal_emb, _ = self.gnn_optimizer.prepare_graph_data(
                            example['semantic_graph'],
                            example['perceptual_graph'],
                            example['goal']
                        )
                        
                        target = torch.tensor(
                            example['optimal_factors'],
                            dtype=torch.float
                        ).to(self.gnn_optimizer.device)
                        
                        pred_factors = self.gnn_optimizer.model(data, goal_emb)
                        loss = criterion(pred_factors, target)
                        
                        val_loss += loss.item()
                        val_count += 1
                    except Exception as e:
                        continue
            
            avg_val_loss = val_loss / val_count if val_count > 0 else float('inf')
            scheduler.step(avg_val_loss)
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1:3d}/{epochs} | "
                      f"Train: {avg_loss:.4f} | Val: {avg_val_loss:.4f}")
            
            # Guardar mejor modelo
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                self.gnn_optimizer.is_trained = True
                self.gnn_optimizer.save_model(save_path)
                if (epoch + 1) % 10 == 0:
                    print(f"  ✓ Mejor modelo guardado (val_loss={best_val_loss:.4f})")
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                print(f"\n⚠️  Early stopping en época {epoch+1}")
                break
        
        print(f"\n{'='*60}")
        print(f"ENTRENAMIENTO COMPLETADO")
        print(f"Mejor Val Loss: {best_val_loss:.4f}")
        print(f"Modelo guardado en: {save_path}")
        print(f"{'='*60}\n")
    
    def _convert_labeled_to_training(self, labeled_example: Dict) -> Dict:
        """Convierte un ejemplo etiquetado al formato de entrenamiento"""
        
        # Crear grafo perceptual
        perceptual_graph = PerceptualGraph()
        
        for obj_name, position in labeled_example['object_positions'].items():
            confidence = np.random.uniform(0.85, 0.95)
            perceptual_graph.add_instance(
                concept=obj_name,
                position=position,
                confidence=confidence,
                bbox=(position[0]-25, position[1]-25, 50, 50)
            )
        
        perceptual_graph.compute_spatial_relations()
        
        semantic_graph = self.semantic_graph.graph
        
        # Crear array de factores óptimos
        optimal_factors = []
        edge_labels = labeled_example.get('edge_labels', {})
        
        for u, v, key, data in semantic_graph.edges(keys=True, data=True):
            action = data.get('action', 'unknown')
            edge_key = (u, action, v)
            
            # Si hay etiqueta explícita, usarla
            if edge_key in edge_labels:
                factor = edge_labels[edge_key]
            else:
                # Si no, usar heurística
                factor = self._compute_heuristic_factor(
                    u, v, action, labeled_example
                )
            
            optimal_factors.append(factor)
        
        return {
            'semantic_graph': semantic_graph,
            'perceptual_graph': perceptual_graph,
            'goal': labeled_example['goal'],
            'optimal_factors': np.array(optimal_factors, dtype=np.float32)
        }
    
    def _compute_heuristic_factor(self, u: str, v: str, action: str, 
                                   scenario: Dict) -> float:
        """Calcula factor heurístico basado en el escenario"""
        target_set = set(scenario.get('target_objects', []))
        required_actions = set(scenario.get('required_actions', []))
        detected_set = set(scenario.get('detected_objects', []))
        
        source_concept = u.split('_')[0] if '_' in u else u
        target_concept = v.split('_')[0] if '_' in v else v
        
        # Heurística 1: Acción requerida hacia target object
        if target_concept in target_set and action in required_actions:
            factor = np.random.uniform(0.15, 0.45)
        # Heurística 2: Acción requerida
        elif action in required_actions:
            factor = np.random.uniform(0.4, 0.8)
        # Heurística 3: Hacia target
        elif target_concept in target_set:
            factor = np.random.uniform(0.6, 1.0)
        # Heurística 4: Objetos detectados y cercanos
        elif target_concept in detected_set and source_concept in detected_set:
            if target_concept in scenario.get('object_positions', {}) and \
               source_concept in scenario.get('object_positions', {}):
                pos1 = np.array(scenario['object_positions'][source_concept])
                pos2 = np.array(scenario['object_positions'][target_concept])
                distance = np.linalg.norm(pos1 - pos2)
                
                if distance < 150:
                    factor = np.random.uniform(0.6, 0.9)
                else:
                    factor = np.random.uniform(0.9, 1.3)
            else:
                factor = np.random.uniform(0.7, 1.2)
        # Heurística 5: Objetos NO detectados
        elif target_concept not in detected_set:
            factor = np.random.uniform(1.2, 1.9)
        # Heurística 6: Irrelevante
        else:
            factor = np.random.uniform(1.3, 1.95)
        
        factor += np.random.normal(0, 0.05)
        return np.clip(factor, 0.1, 2.0)
    
    def generate_synthetic_training_data(self, n_examples: int = 50) -> List[Dict]:
        """
        Genera datos de entrenamiento sintéticos basados en heurísticas.
        Usa los mismos escenarios del código original.
        """
        print(f"\n{'='*60}")
        print(f"Generando {n_examples} ejemplos sintéticos de entrenamiento...")
        print(f"{'='*60}")
        
        training_examples = []
        
        # Escenarios básicos (versión reducida del original)
        scenarios = [
            {
                'goal': 'Move the cup to the table',
                'target_objects': ['cup', 'table'],
                'required_actions': ['grasp', 'move'],
                'detected_objects': ['cup', 'table', 'chair', 'bowl'],
                'object_positions': {
                    'cup': (200, 250, 1.1),
                    'table': (350, 300, 1.4),
                    'chair': (280, 350, 1.5),
                    'bowl': (420, 280, 1.3)
                }
            },
            {
                'goal': 'Open the cabinet to access utensils',
                'target_objects': ['cabinet'],
                'required_actions': ['open'],
                'detected_objects': ['cabinet', 'table', 'drawer', 'chair'],
                'object_positions': {
                    'cabinet': (320, 300, 1.5),
                    'table': (350, 320, 1.6),
                    'drawer': (300, 280, 1.3),
                    'chair': (280, 350, 1.7)
                }
            },
            {
                'goal': 'Inspect the bottle to check contents',
                'target_objects': ['bottle'],
                'required_actions': ['inspect'],
                'detected_objects': ['bottle', 'table', 'cup', 'bowl'],
                'object_positions': {
                    'bottle': (290, 270, 1.1),
                    'table': (300, 320, 1.4),
                    'cup': (340, 280, 1.15),
                    'bowl': (380, 290, 1.2)
                }
            },
            {
                'goal': 'Move the vase to the shelf',
                'target_objects': ['vase', 'shelf'],
                'required_actions': ['grasp', 'move'],
                'detected_objects': ['shelf', 'book', 'clock'],  # vase NO detectado
                'object_positions': {
                    'shelf': (400, 320, 1.6),
                    'book': (280, 290, 1.1),
                    'clock': (420, 300, 1.55)
                }
            },
        ]
        
        # Generar múltiples ejemplos por escenario
        examples_per_scenario = max(1, n_examples // len(scenarios))
        
        for scenario in scenarios:
            for _ in range(examples_per_scenario):
                example = self._convert_labeled_to_training(scenario)
                if example:
                    training_examples.append(example)
        
        print(f"✓ Generados {len(training_examples)} ejemplos de entrenamiento")
        return training_examples
    
    def train(self, epochs: int = 200, batch_size: int = 8, 
              validation_split: float = 0.2, 
              early_stopping_patience: int = 20,   
              save_path: str = "gnn_cost_optimizer.pth"):
        """
        Entrena el modelo GNN con datos sintéticos generados automáticamente.
        
        Args:
            epochs: Número de épocas de entrenamiento
            batch_size: Número de ejemplos a generar
            validation_split: Proporción para validación
            early_stopping_patience: Paciencia para early stopping
            save_path: Ruta donde guardar el modelo
        """
        print(f"\n{'='*60}")
        print(f"INICIANDO ENTRENAMIENTO GNN")
        print(f"{'='*60}")
        print(f"Epochs: {epochs}")
        print(f"Examples: {batch_size * 20}")
        print(f"Device: {self.gnn_optimizer.device}")
        print(f"{'='*60}\n")
        
        # Generar datos de entrenamiento sintéticos
        training_data = self.generate_synthetic_training_data(n_examples=batch_size * 20)
        
        if not training_data:
            print("❌ Error: No se generaron datos de entrenamiento")
            return
        
        # Dividir en train/val
        n_val = int(len(training_data) * validation_split)
        np.random.shuffle(training_data)
        val_data = training_data[:n_val]
        train_data = training_data[n_val:]
        
        print(f"Train: {len(train_data)} ejemplos | Val: {len(val_data)} ejemplos\n")
        
        # Ajustar normalizadores
        sample_goals = [ex['goal'] for ex in training_data[:min(10, len(training_data))]]
        self.gnn_optimizer.feature_extractor.fit_scalers(
            self.semantic_graph.graph,
            train_data[0]['perceptual_graph'],
            sample_goals
        )
        
        # Inicializar modelo si no existe
        if self.gnn_optimizer.model is None:
            first_example = training_data[0]
            self.gnn_optimizer.initialize_model(
                first_example['semantic_graph'],
                first_example['perceptual_graph'],
                first_example['goal']
            )
        
        # Configurar optimización
        optimizer = torch.optim.AdamW(
            self.gnn_optimizer.model.parameters(), 
            lr=0.001,
            weight_decay=0.01
        )
        criterion = WeightedHuberLoss(delta=0.15)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Loop de entrenamiento
        for epoch in range(epochs):
            self.gnn_optimizer.model.train()            
            total_loss = 0
            batch_count = 0
            
            np.random.shuffle(train_data)
            
            for example in train_data:
                optimizer.zero_grad()
                
                try:
                    data, goal_emb, _ = self.gnn_optimizer.prepare_graph_data(
                        example['semantic_graph'],
                        example['perceptual_graph'],
                        example['goal']
                    )
                    
                    target = torch.tensor(
                        example['optimal_factors'],
                        dtype=torch.float
                    ).to(self.gnn_optimizer.device)
                    
                    pred_factors = self.gnn_optimizer.model(data, goal_emb)
                    loss = criterion(pred_factors, target)
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.gnn_optimizer.model.parameters(), 
                        max_norm=1.0
                    )
                    optimizer.step()
                    
                    total_loss += loss.item()
                    batch_count += 1
                    
                except Exception as e:
                    print(f"❌ Error en ejemplo: {e}")
                    continue
            
            avg_loss = total_loss / batch_count if batch_count > 0 else float('inf')
            
            # Validación
            self.gnn_optimizer.model.eval()
            val_loss = 0
            val_count = 0
            
            with torch.no_grad():
                for example in val_data:
                    try:
                        data, goal_emb, _ = self.gnn_optimizer.prepare_graph_data(
                            example['semantic_graph'],
                            example['perceptual_graph'],
                            example['goal']
                        )
                        
                        target = torch.tensor(
                            example['optimal_factors'],
                            dtype=torch.float
                        ).to(self.gnn_optimizer.device)
                        
                        pred_factors = self.gnn_optimizer.model(data, goal_emb)
                        loss = criterion(pred_factors, target)
                        
                        val_loss += loss.item()
                        val_count += 1
                        
                    except Exception as e:
                        continue
            
            avg_val_loss = val_loss / val_count if val_count > 0 else float('inf')
            scheduler.step(avg_val_loss)
            
            # Log progreso
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1:3d}/{epochs} | Train: {avg_loss:.4f} | Val: {avg_val_loss:.4f}")
            
            # Guardar mejor modelo
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                self.gnn_optimizer.is_trained = True
                self.gnn_optimizer.save_model(save_path)
                if (epoch + 1) % 10 == 0:
                    print(f"  ✓ Nuevo mejor modelo guardado (loss={best_val_loss:.4f})")
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                print(f"\n Early stopping en época {epoch+1}")
                print(f"   No hay mejora en validation loss por {early_stopping_patience} épocas.")
                break
        
        print(f"\n{'='*60}")
        print(f"ENTRENAMIENTO COMPLETADO")
        print(f"{'='*60}")
        print(f"Train Loss final: {avg_loss:.4f}")
        print(f"Val Loss final: {avg_val_loss:.4f}")
        print(f"Mejor Val Loss: {best_val_loss:.4f}")
        print(f"Modelo guardado en: {save_path}")
        print(f"{'='*60}\n")