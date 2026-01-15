import json
import numpy as np
from typing import Dict, List, Any, Tuple
from sklearn.metrics import (
    precision_recall_fscore_support, accuracy_score,
    confusion_matrix, classification_report
)
import pandas as pd
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class TextModelEvaluator:
    """Evaluator for text models"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.results_dir = Path("results/text_models")
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def evaluate_ner_model(self, predictions: List[Dict], 
                          ground_truth: List[Dict]) -> Dict[str, Any]:
        """Evaluate NER model performance"""
        # Flatten predictions
        pred_entities = []
        true_entities = []
        
        for pred, true in zip(predictions, ground_truth):
            pred_entities.extend(pred.get("entities", []))
            true_entities.extend(true.get("entities", []))
        
        # Calculate metrics
        metrics = self._calculate_ner_metrics(pred_entities, true_entities)
        
        # Generate detailed report
        report = self._generate_ner_report(predictions, ground_truth)
        
        # Visualize results
        self._plot_ner_results(metrics, report)
        
        return {
            "metrics": metrics,
            "report": report,
            "confusion_matrix": self._create_ner_confusion_matrix(pred_entities, true_entities)
        }
    
    def _calculate_ner_metrics(self, pred_entities, true_entities) -> Dict:
        """Calculate NER-specific metrics"""
        # Convert to set for comparison
        pred_set = set(pred_entities)
        true_set = set(true_entities)
        
        # Calculate precision, recall, F1
        tp = len(pred_set.intersection(true_set))
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "support": len(true_set)
        }
    
    def evaluate_topic_segmentation(self, predicted_boundaries: List[float],
                                   true_boundaries: List[float]) -> Dict[str, Any]:
        """Evaluate topic segmentation accuracy"""
        tolerance = self.config['evaluation']['topic_boundary_tolerance']
        
        # Match boundaries within tolerance
        matched = 0
        for pred in predicted_boundaries:
            for true in true_boundaries:
                if abs(pred - true) <= tolerance:
                    matched += 1
                    break
        
        precision = matched / len(predicted_boundaries) if predicted_boundaries else 0
        recall = matched / len(true_boundaries) if true_boundaries else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate boundary deviation
        deviations = []
        for pred in predicted_boundaries:
            closest_true = min(true_boundaries, key=lambda x: abs(x - pred))
            deviations.append(abs(pred - closest_true))
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "detected_boundaries": len(predicted_boundaries),
            "true_boundaries": len(true_boundaries),
            "matched_boundaries": matched,
            "avg_deviation": np.mean(deviations) if deviations else 0,
            "max_deviation": np.max(deviations) if deviations else 0
        }
    
    def evaluate_similarity_model(self, embeddings: np.ndarray,
                                 similarity_matrix: np.ndarray,
                                 true_similarity: np.ndarray) -> Dict[str, Any]:
        """Evaluate similarity model performance"""
        # Calculate correlation with true similarity
        correlation, p_value = pearsonr(
            similarity_matrix.flatten(),
            true_similarity.flatten()
        )
        
        # Calculate clustering metrics
        clustering_metrics = self._calculate_clustering_metrics(embeddings)
        
        # Calculate retrieval metrics
        retrieval_metrics = self._calculate_retrieval_metrics(embeddings, true_similarity)
        
        return {
            "correlation": {
                "pearson": correlation,
                "p_value": p_value
            },
            "clustering": clustering_metrics,
            "retrieval": retrieval_metrics,
            "embedding_statistics": {
                "mean": np.mean(embeddings),
                "std": np.std(embeddings),
                "shape": embeddings.shape
            }
        }
    
    def _calculate_clustering_metrics(self, embeddings: np.ndarray) -> Dict:
        """Calculate clustering quality metrics"""
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score, calinski_harabasz_score
        
        # Try different cluster numbers
        n_samples = len(embeddings)
        n_clusters_range = range(2, min(10, n_samples // 10))
        
        results = {}
        for n_clusters in n_clusters_range:
            if n_samples > n_clusters:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                labels = kmeans.fit_predict(embeddings)
                
                if len(set(labels)) > 1:  # Need at least 2 clusters
                    silhouette = silhouette_score(embeddings, labels)
                    calinski = calinski_harabasz_score(embeddings, labels)
                    
                    results[n_clusters] = {
                        "silhouette": silhouette,
                        "calinski_harabasz": calinski,
                        "inertia": kmeans.inertia_
                    }
        
        return results
    
    def _calculate_retrieval_metrics(self, embeddings: np.ndarray,
                                    true_similarity: np.ndarray) -> Dict:
        """Calculate retrieval metrics"""
        from sklearn.neighbors import NearestNeighbors
        
        n_neighbors = min(5, len(embeddings) - 1)
        
        # Fit nearest neighbors
        nn = NearestNeighbors(n_neighbors=n_neighbors)
        nn.fit(embeddings)
        
        # Get neighbors
        distances, indices = nn.kneighbors(embeddings)
        
        # Calculate precision@k
        precision_at_k = []
        for i, neighbor_indices in enumerate(indices):
            # Skip self
            neighbor_indices = neighbor_indices[1:]
            
            # Calculate similarity with true neighbors
            similarities = true_similarity[i, neighbor_indices]
            precision_at_k.append(np.mean(similarities > self.config['sentence_bert']['similarity_threshold']))
        
        return {
            "precision_at_k": np.mean(precision_at_k),
            "avg_distance": np.mean(distances),
            "n_neighbors": n_neighbors
        }
    
    def _generate_ner_report(self, predictions: List[Dict],
                            ground_truth: List[Dict]) -> pd.DataFrame:
        """Generate detailed NER report"""
        reports = []
        
        for pred, true in zip(predictions, ground_truth):
            pred_text = pred.get("text", "")
            true_text = true.get("text", "")
            
            if pred_text == true_text:
                pred_ents = pred.get("entities", [])
                true_ents = true.get("entities", [])
                
                report = {
                    "text": pred_text[:100] + "..." if len(pred_text) > 100 else pred_text,
                    "predicted_entities": len(pred_ents),
                    "true_entities": len(true_ents),
                    "correct_entities": len(set(pred_ents).intersection(set(true_ents))),
                    "precision": len(set(pred_ents).intersection(set(true_ents))) / len(pred_ents) if pred_ents else 0,
                    "recall": len(set(pred_ents).intersection(set(true_ents))) / len(true_ents) if true_ents else 0
                }
                reports.append(report)
        
        return pd.DataFrame(reports)
    
    def _plot_ner_results(self, metrics: Dict, report: pd.DataFrame):
        """Plot NER evaluation results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Precision, Recall, F1 bar plot
        metrics_values = [metrics['precision'], metrics['recall'], metrics['f1']]
        metrics_labels = ['Precision', 'Recall', 'F1']
        axes[0, 0].bar(metrics_labels, metrics_values, color=['blue', 'green', 'red'])
        axes[0, 0].set_title('NER Performance Metrics')
        axes[0, 0].set_ylim([0, 1])
        
        # Entity count distribution
        axes[0, 1].hist(report['predicted_entities'], alpha=0.5, label='Predicted', bins=10)
        axes[0, 1].hist(report['true_entities'], alpha=0.5, label='True', bins=10)
        axes[0, 1].set_title('Entity Count Distribution')
        axes[0, 1].legend()
        
        # Precision vs Recall scatter
        axes[1, 0].scatter(report['precision'], report['recall'], alpha=0.5)
        axes[1, 0].set_xlabel('Precision')
        axes[1, 0].set_ylabel('Recall')
        axes[1, 0].set_title('Precision vs Recall per Segment')
        axes[1, 0].set_xlim([0, 1])
        axes[1, 0].set_ylim([0, 1])
        
        # Text length vs entities
        text_lengths = report['text'].apply(len)
        axes[1, 1].scatter(text_lengths, report['predicted_entities'], alpha=0.5, label='Predicted')
        axes[1, 1].scatter(text_lengths, report['true_entities'], alpha=0.5, label='True')
        axes[1, 1].set_xlabel('Text Length')
        axes[1, 1].set_ylabel('Entity Count')
        axes[1, 1].set_title('Text Length vs Entity Count')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'ner_evaluation.png', dpi=300)
        plt.close()
    
    def save_evaluation_results(self, results: Dict, model_name: str):
        """Save evaluation results to file"""
        results_file = self.results_dir / f"{model_name}_evaluation.json"
        
        # Convert numpy types to Python types
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            else:
                return obj
        
        results = convert_types(results)
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Evaluation results saved to {results_file}")
        return results_file