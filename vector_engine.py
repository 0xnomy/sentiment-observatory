import json
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv
from groq import Groq

try:
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.decomposition import PCA
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

MODEL_NAME = "meta-llama/llama-4-maverick-17b-128e-instruct"


class VectorSearchEngine:
    def __init__(self, data_dir: str = "data/vectors"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Vector data files
        self.embeddings_file = self.data_dir / "content_embeddings.json"
        self.clusters_file = self.data_dir / "content_clusters.json"
        self.similarity_index_file = self.data_dir / "similarity_index.json"
        
        # Load existing data
        self.content_embeddings = self._load_json(self.embeddings_file, {})
        self.content_clusters = self._load_json(self.clusters_file, {})
        self.similarity_index = self._load_json(self.similarity_index_file, {})
        
        # Vector parameters
        self.embedding_dimension = 1536  # Default for text embeddings
        self.similarity_threshold = 0.7
        self.max_cluster_size = 50
        
        # Initialize ML components if available
        if ML_AVAILABLE:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            self.pca = PCA(n_components=min(50, self.embedding_dimension))
    
    def _load_json(self, file_path: Path, default_value: Any) -> Any:
        try:
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Validate data structure for similarity index
                    if file_path.name == "similarity_index.json" and isinstance(data, dict):
                        # Clean up any non-numeric similarity values
                        cleaned_data = {}
                        for content_id, similarities in data.items():
                            if isinstance(similarities, dict):
                                cleaned_similarities = {}
                                for similar_id, similarity_score in similarities.items():
                                    if isinstance(similarity_score, (int, float)):
                                        cleaned_similarities[similar_id] = similarity_score
                                    else:
                                        print(f"Warning: Invalid similarity score type for {content_id} -> {similar_id}: {type(similarity_score)}")
                                if cleaned_similarities:
                                    cleaned_data[content_id] = cleaned_similarities
                        return cleaned_data
                    return data
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
        return default_value
    
    def _save_json(self, file_path: Path, data: Any) -> None:
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving to {file_path}: {e}")
    
    def generate_text_embedding(self, text: str) -> List[float]:
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("Missing GROQ_API_KEY in environment.")
        
        client = Groq(api_key=api_key)
        
        prompt = f"Convert the following text into a numerical representation for similarity analysis. Focus on key themes, sentiment, and content type:\n\n{text[:1000]}"
        
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a text analysis assistant. Provide a concise summary of the key themes and characteristics of the given text."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=200
            )
            
            summary = response.choices[0].message.content if response.choices else ""
            
            # Create a simple feature vector based on text characteristics
            embedding = self._create_feature_vector(text, summary)
            return embedding
            
        except Exception as e:
            print(f"Error generating embedding: {e}")
            # Fallback to basic feature vector
            return self._create_feature_vector(text, "")
    
    def _create_feature_vector(self, text: str, summary: str) -> List[float]:
        """Create a feature vector from text characteristics."""
        features = []
        
        # Text length features
        features.append(min(len(text) / 1000, 1.0))  # Normalized length
        
        # Word count features
        words = text.split()
        features.append(min(len(words) / 100, 1.0))  # Normalized word count
        
        # Sentiment indicators
        positive_words = ["good", "great", "excellent", "amazing", "wonderful", "love", "like"]
        negative_words = ["bad", "terrible", "awful", "hate", "dislike", "horrible", "worst"]
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        features.append(min(positive_count / 10, 1.0))
        features.append(min(negative_count / 10, 1.0))
        
        # Question indicators
        question_count = text.count('?')
        features.append(min(question_count / 5, 1.0))
        
        # Exclamation indicators
        exclamation_count = text.count('!')
        features.append(min(exclamation_count / 5, 1.0))
        
        # URL indicators
        url_count = len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text))
        features.append(min(url_count / 3, 1.0))
        
        # Hashtag indicators
        hashtag_count = len(re.findall(r'#\w+', text))
        features.append(min(hashtag_count / 5, 1.0))
        
        # Mention indicators
        mention_count = len(re.findall(r'@\w+', text))
        features.append(min(mention_count / 5, 1.0))
        
        # Number indicators
        number_count = len(re.findall(r'\d+', text))
        features.append(min(number_count / 10, 1.0))
        
        # Capitalization features
        capital_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        features.append(capital_ratio)
        
        # Fill remaining dimensions with zeros or random values
        while len(features) < self.embedding_dimension:
            features.append(0.0)
        
        # Normalize to unit vector
        features = features[:self.embedding_dimension]
        norm = np.linalg.norm(features)
        if norm > 0:
            features = [f / norm for f in features]
        
        return features
    
    def add_content_embedding(self, content_id: str, content_data: Dict[str, Any]) -> None:
        """Add content embedding to the vector store."""
        # Extract text for embedding
        text = content_data.get("text", "") or content_data.get("body", "") or content_data.get("summary", "")
        
        if not text:
            return
        
        # Generate embedding
        embedding = self.generate_text_embedding(text)
        
        # Store embedding with metadata
        self.content_embeddings[content_id] = {
            "embedding": embedding,
            "metadata": {
                "platform": content_data.get("platform", "unknown"),
                "themes": content_data.get("theme_categorization", []),
                "sentiment": content_data.get("sentiment", {}).get("label", "neutral"),
                "virality_score": content_data.get("virality_score", 5),
                "timestamp": datetime.now().isoformat(),
                "text_length": len(text),
                "url": content_data.get("url", "")
            }
        }
        
        # Save updated embeddings
        self._save_json(self.embeddings_file, self.content_embeddings)
        
        # Update similarity index
        self._update_similarity_index(content_id, embedding)
    
    def _update_similarity_index(self, new_content_id: str, new_embedding: List[float]) -> None:
        """Update similarity index with new content."""
        if new_content_id not in self.similarity_index:
            self.similarity_index[new_content_id] = {}
        
        # Calculate similarity with existing content
        for existing_id, existing_data in self.content_embeddings.items():
            if existing_id == new_content_id:
                continue
            
            existing_embedding = existing_data["embedding"]
            similarity = self._calculate_cosine_similarity(new_embedding, existing_embedding)
            
            # Debug logging
            if not isinstance(similarity, (int, float)):
                print(f"Warning: Invalid similarity type for {new_content_id} -> {existing_id}: {type(similarity)} = {similarity}")
                continue
            
            # Store bidirectional similarity
            if similarity > self.similarity_threshold:
                self.similarity_index[new_content_id][existing_id] = similarity
                
                if existing_id not in self.similarity_index:
                    self.similarity_index[existing_id] = {}
                self.similarity_index[existing_id][new_content_id] = similarity
        
        # Save updated similarity index
        self._save_json(self.similarity_index_file, self.similarity_index)
    
    def _calculate_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def find_similar_content(self, content_id: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Find most similar content to a given content ID."""
        if content_id not in self.similarity_index:
            return []
        
        # Get similarities for this content
        similarities = self.similarity_index[content_id]
        
        # Sort by similarity score
        sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        
        # Get top k similar content
        similar_content = []
        for similar_id, similarity_score in sorted_similarities[:top_k]:
            if similar_id in self.content_embeddings:
                similar_content.append({
                    "content_id": similar_id,
                    "similarity_score": round(similarity_score, 4),
                    "metadata": self.content_embeddings[similar_id]["metadata"]
                })
        
        return similar_content
    
    def semantic_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Perform semantic search using a text query."""
        # Generate embedding for query
        query_embedding = self.generate_text_embedding(query)
        
        # Calculate similarity with all content
        similarities = []
        for content_id, content_data in self.content_embeddings.items():
            similarity = self._calculate_cosine_similarity(query_embedding, content_data["embedding"])
            if isinstance(similarity, (int, float)):
                similarities.append((content_id, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k results
        search_results = []
        for content_id, similarity_score in similarities[:top_k]:
            if isinstance(similarity_score, (int, float)) and similarity_score > 0.1:  # Minimum similarity threshold
                search_results.append({
                    "content_id": content_id,
                    "similarity_score": round(similarity_score, 4),
                    "metadata": self.content_embeddings[content_id]["metadata"]
                })
        
        return search_results
    
    def cluster_content(self, min_cluster_size: int = 3) -> Dict[str, Any]:
        """Cluster content based on embeddings."""
        if not ML_AVAILABLE:
            return {"error": "Machine learning libraries not available"}
        
        if len(self.content_embeddings) < min_cluster_size:
            return {"error": f"Not enough content for clustering (need {min_cluster_size}, have {len(self.content_embeddings)})"}
        
        # Extract embeddings and metadata
        content_ids = list(self.content_embeddings.keys())
        embeddings_matrix = np.array([self.content_embeddings[cid]["embedding"] for cid in content_ids])
        
        # Reduce dimensionality for clustering
        if embeddings_matrix.shape[1] > 50:
            embeddings_reduced = self.pca.fit_transform(embeddings_matrix)
        else:
            embeddings_reduced = embeddings_matrix
        
        # Perform clustering
        try:
            # Try DBSCAN first for automatic cluster detection
            clustering = DBSCAN(eps=0.3, min_samples=min_cluster_size)
            cluster_labels = clustering.fit_predict(embeddings_reduced)
            
            # If DBSCAN doesn't find good clusters, try K-means
            if len(set(cluster_labels)) < 2 or -1 in cluster_labels:
                n_clusters = min(10, len(content_ids) // min_cluster_size)
                clustering = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = clustering.fit_predict(embeddings_reduced)
        except Exception as e:
            return {"error": f"Clustering failed: {str(e)}"}
        
        # Organize clusters
        clusters = defaultdict(list)
        for i, label in enumerate(cluster_labels):
            content_id = content_ids[i]
            clusters[f"cluster_{label}"].append({
                "content_id": content_id,
                "metadata": self.content_embeddings[content_id]["metadata"]
            })
        
        # Calculate cluster statistics
        cluster_stats = {}
        for cluster_id, cluster_content in clusters.items():
            if len(cluster_content) >= min_cluster_size:
                # Calculate cluster centroid
                cluster_embeddings = [self.content_embeddings[item["content_id"]]["embedding"] for item in cluster_content]
                centroid = np.mean(cluster_embeddings, axis=0)
                
                # Calculate cluster cohesion
                cohesion = np.mean([
                    self._calculate_cosine_similarity(centroid, emb) 
                    for emb in cluster_embeddings
                ])
                
                # Extract common themes
                all_themes = []
                for item in cluster_content:
                    themes = item["metadata"].get("themes", [])
                    all_themes.extend([t["theme"] for t in themes])
                
                theme_counts = defaultdict(int)
                for theme in all_themes:
                    theme_counts[theme] += 1
                
                common_themes = [theme for theme, count in theme_counts.items() if count > 1]
                
                cluster_stats[cluster_id] = {
                    "size": len(cluster_content),
                    "cohesion": round(cohesion, 4),
                    "common_themes": common_themes,
                    "avg_sentiment": self._calculate_cluster_sentiment(cluster_content),
                    "avg_virality": self._calculate_cluster_virality(cluster_content),
                    "platforms": list(set(item["metadata"]["platform"] for item in cluster_content))
                }
        
        # Save clusters
        self.content_clusters = {
            "clusters": dict(clusters),
            "statistics": cluster_stats,
            "clustering_method": "DBSCAN+KMeans",
            "timestamp": datetime.now().isoformat()
        }
        
        self._save_json(self.clusters_file, self.content_clusters)
        
        return self.content_clusters
    
    def _calculate_cluster_sentiment(self, cluster_content: List[Dict[str, Any]]) -> str:
        """Calculate average sentiment for a cluster."""
        sentiment_counts = defaultdict(int)
        for item in cluster_content:
            sentiment = item["metadata"].get("sentiment", "neutral")
            sentiment_counts[sentiment] += 1
        
        if sentiment_counts:
            return max(sentiment_counts.items(), key=lambda x: x[1])[0]
        return "neutral"
    
    def _calculate_cluster_virality(self, cluster_content: List[Dict[str, Any]]) -> float:
        """Calculate average virality score for a cluster."""
        virality_scores = [
            item["metadata"].get("virality_score", 5) 
            for item in cluster_content
        ]
        
        if virality_scores:
            return round(np.mean(virality_scores), 2)
        return 5.0
    
    def find_content_clusters(self, content_id: str) -> List[Dict[str, Any]]:
        """Find which clusters a piece of content belongs to."""
        if not self.content_clusters:
            return []
        
        clusters_found = []
        for cluster_id, cluster_content in self.content_clusters["clusters"].items():
            for item in cluster_content:
                if item["content_id"] == content_id:
                    clusters_found.append({
                        "cluster_id": cluster_id,
                        "cluster_size": len(cluster_content),
                        "cluster_stats": self.content_clusters["statistics"].get(cluster_id, {}),
                        "cluster_members": cluster_content
                    })
                    break
        
        return clusters_found
    
    def generate_vector_report(self) -> Dict[str, Any]:
        """Generate comprehensive vector analysis report."""
        total_content = len(self.content_embeddings)
        
        if total_content == 0:
            return {"error": "No content embeddings available"}
        
        # Calculate embedding statistics
        embedding_dimensions = len(next(iter(self.content_embeddings.values()))["embedding"])
        
        # Analyze similarity distribution
        similarity_scores = []
        for content_id, similarities in self.similarity_index.items():
            similarity_scores.extend(similarities.values())
        
        avg_similarity = np.mean(similarity_scores) if similarity_scores else 0
        
        # Platform distribution
        platform_counts = defaultdict(int)
        for content_data in self.content_embeddings.values():
            platform = content_data["metadata"]["platform"]
            platform_counts[platform] += 1
        
        # Theme distribution
        theme_counts = defaultdict(int)
        for content_data in self.content_embeddings.values():
            themes = content_data["metadata"].get("themes", [])
            for theme in themes:
                theme_counts[theme["theme"]] += 1
        
        # Clustering status
        clustering_status = "completed" if self.content_clusters else "not_performed"
        cluster_count = len(self.content_clusters.get("clusters", {})) if self.content_clusters else 0
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_content": total_content,
                "embedding_dimensions": embedding_dimensions,
                "clustering_status": clustering_status,
                "cluster_count": cluster_count
            },
            "statistics": {
                "average_similarity": round(avg_similarity, 4),
                "similarity_pairs": len(similarity_scores),
                "platform_distribution": dict(platform_counts),
                "top_themes": sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            },
            "clusters": self.content_clusters if self.content_clusters else {},
            "recommendations": self._generate_vector_recommendations()
        }
        
        return report
    
    def _generate_vector_recommendations(self) -> List[str]:
        """Generate recommendations based on vector analysis."""
        recommendations = []
        
        total_content = len(self.content_embeddings)
        
        if total_content < 10:
            recommendations.append("Need more content for meaningful vector analysis")
            return recommendations
        
        # Similarity analysis
        if self.similarity_index:
            high_similarity_count = sum(
                1 for similarities in self.similarity_index.values()
                for score in similarities.values()
                if isinstance(score, (int, float)) and score > 0.8
            )
            
            if high_similarity_count > total_content * 0.1:
                recommendations.append("High content similarity detected - consider content diversification")
            elif high_similarity_count < total_content * 0.01:
                recommendations.append("Low content similarity - content is highly diverse")
        
        # Clustering recommendations
        if self.content_clusters:
            cluster_sizes = [len(cluster) for cluster in self.content_clusters["clusters"].values()]
            avg_cluster_size = np.mean(cluster_sizes)
            
            if avg_cluster_size > 20:
                recommendations.append("Large clusters detected - consider sub-clustering for better organization")
            elif avg_cluster_size < 5:
                recommendations.append("Small clusters detected - may need more content for stable clustering")
        
        # Platform recommendations
        platform_counts = defaultdict(int)
        for content_data in self.content_embeddings.values():
            platform = content_data["metadata"]["platform"]
            platform_counts[platform] += 1
        
        if len(platform_counts) < 2:
            recommendations.append("Limited platform diversity - consider expanding to more platforms")
        
        return recommendations[:5]  # Top 5 recommendations


def analyze_content_vectors(content_data: Dict[str, Any], vector_engine: VectorSearchEngine) -> Dict[str, Any]:
    """Analyze vectors for a single piece of content."""
    # Generate content ID
    content_id = content_data.get("url", "") or f"content_{datetime.now().timestamp()}"
    
    # Add content embedding
    vector_engine.add_content_embedding(content_id, content_data)
    
    # Find similar content
    similar_content = vector_engine.find_similar_content(content_id, top_k=5)
    
    # Find content clusters
    content_clusters = vector_engine.find_content_clusters(content_id)
    
    return {
        "content_id": content_id,
        "embedding_generated": True,
        "similar_content_count": len(similar_content),
        "similar_content": similar_content,
        "cluster_membership": len(content_clusters),
        "clusters": content_clusters
    }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python vector_engine.py <command> [args...]", file=sys.stderr)
        print("Commands: search, similar, cluster, report", file=sys.stderr)
        raise SystemExit(1)
    
    command = sys.argv[1]
    vector_engine = VectorSearchEngine()
    
    if command == "search":
        if len(sys.argv) < 3:
            print("Usage: python vector_engine.py search <query>", file=sys.stderr)
            raise SystemExit(1)
        query = sys.argv[2]
        results = vector_engine.semantic_search(query)
        print(json.dumps(results, ensure_ascii=False, indent=2))
    elif command == "similar":
        if len(sys.argv) < 3:
            print("Usage: python vector_engine.py similar <content_id>", file=sys.stderr)
            raise SystemExit(1)
        content_id = sys.argv[2]
        results = vector_engine.find_similar_content(content_id)
        print(json.dumps(results, ensure_ascii=False, indent=2))
    elif command == "cluster":
        results = vector_engine.cluster_content()
        print(json.dumps(results, ensure_ascii=False, indent=2))
    elif command == "report":
        report = vector_engine.generate_vector_report()
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"Unknown command: {command}", file=sys.stderr)
        raise SystemExit(1)
