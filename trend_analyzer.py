import json
import os
import re
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv
from groq import Groq

MODEL_NAME = "meta-llama/llama-4-maverick-17b-128e-instruct"


class TrendAnalyzer:
    def __init__(self, data_dir: str = "data/trends"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.trends_file = self.data_dir / "trends_history.json"
        self.themes_file = self.data_dir / "themes_history.json"
        self.sentiment_file = self.data_dir / "sentiment_history.json"
        
        # Load existing data
        self.trends_history = self._load_json(self.trends_file, [])
        self.themes_history = self._load_json(self.themes_file, {})
        self.sentiment_history = self._load_json(self.sentiment_file, {})
        
        # Trend detection parameters
        self.min_volume_threshold = 3  # Minimum mentions to consider trending
        self.trend_window_hours = 24   # Time window for trend detection
        self.velocity_threshold = 0.5   # Minimum velocity for trend detection
        
    def _load_json(self, file_path: Path, default_value: Any) -> Any:
        """Load JSON data from file with fallback to default value."""
        try:
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception:
            pass
        return default_value
    
    def _save_json(self, file_path: Path, data: Any) -> None:
        """Save data to JSON file."""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving to {file_path}: {e}")
    
    def add_content(self, content_data: Dict[str, Any]) -> None:
        """Add new content to the trend analysis system."""
        timestamp = datetime.now()
        
        # Extract key information
        platform = content_data.get("platform", "unknown")
        themes = content_data.get("theme_categorization", [])
        sentiment = content_data.get("sentiment", {}).get("label", "neutral")
        virality_score = content_data.get("virality_score", 5)
        
        # Add to trends history
        trend_entry = {
            "timestamp": timestamp.isoformat(),
            "platform": platform,
            "themes": [t["theme"] for t in themes],
            "sentiment": sentiment,
            "virality_score": virality_score,
            "url": content_data.get("url", ""),
            "summary": content_data.get("summary", "")[:200]
        }
        
        self.trends_history.append(trend_entry)
        
        # Update themes history
        for theme in themes:
            theme_name = theme["theme"]
            if theme_name not in self.themes_history:
                self.themes_history[theme_name] = []
            
            self.themes_history[theme_name].append({
                "timestamp": timestamp.isoformat(),
                "platform": platform,
                "sentiment": sentiment,
                "virality_score": virality_score,
                "confidence": theme.get("confidence", 0.5)
            })
        
        # Update sentiment history
        if sentiment not in self.sentiment_history:
            self.sentiment_history[sentiment] = []
        
        self.sentiment_history[sentiment].append({
            "timestamp": timestamp.isoformat(),
            "platform": platform,
            "themes": [t["theme"] for t in themes],
            "virality_score": virality_score
        })
        
        # Save updated data
        self._save_json(self.trends_file, self.trends_history)
        self._save_json(self.themes_file, self.themes_history)
        self._save_json(self.sentiment_file, self.sentiment_history)
    
    def detect_trending_topics(self, hours_back: int = 24) -> List[Dict[str, Any]]:
        """Detect currently trending topics based on recent activity."""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        # Filter recent trends
        recent_trends = [
            t for t in self.trends_history 
            if datetime.fromisoformat(t["timestamp"]) > cutoff_time
        ]
        
        # Count theme mentions
        theme_counts = Counter()
        theme_sentiments = defaultdict(list)
        theme_virality = defaultdict(list)
        
        for trend in recent_trends:
            for theme in trend["themes"]:
                theme_counts[theme] += 1
                theme_sentiments[theme].append(trend["sentiment"])
                theme_virality[theme].append(trend["virality_score"])
        
        # Calculate trend metrics
        trending_topics = []
        for theme, count in theme_counts.items():
            if count >= self.min_volume_threshold:
                # Calculate velocity (mentions per hour)
                velocity = count / hours_back
                
                # Calculate sentiment distribution
                sentiments = theme_sentiments[theme]
                sentiment_dist = Counter(sentiments)
                dominant_sentiment = sentiment_dist.most_common(1)[0][0] if sentiment_dist else "neutral"
                
                # Calculate average virality
                avg_virality = np.mean(theme_virality[theme]) if theme_virality[theme] else 5
                
                # Calculate momentum (acceleration)
                momentum = self._calculate_momentum(theme, hours_back)
                
                trending_topics.append({
                    "theme": theme,
                    "mention_count": count,
                    "velocity": round(velocity, 2),
                    "momentum": round(momentum, 2),
                    "dominant_sentiment": dominant_sentiment,
                    "sentiment_distribution": dict(sentiment_dist),
                    "average_virality": round(avg_virality, 2),
                    "trend_strength": self._calculate_trend_strength(count, velocity, momentum)
                })
        
        # Sort by trend strength
        trending_topics.sort(key=lambda x: x["trend_strength"], reverse=True)
        return trending_topics[:20]  # Top 20 trending topics
    
    def _calculate_momentum(self, theme: str, hours_back: int) -> float:
        """Calculate momentum (acceleration) for a theme."""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        # Split time window into two halves
        mid_time = datetime.now() - timedelta(hours=hours_back/2)
        
        # Count mentions in first and second half
        first_half = sum(
            1 for t in self.themes_history.get(theme, [])
            if cutoff_time < datetime.fromisoformat(t["timestamp"]) <= mid_time
        )
        
        second_half = sum(
            1 for t in self.themes_history.get(theme, [])
            if mid_time < datetime.fromisoformat(t["timestamp"]) <= datetime.now()
        )
        
        # Momentum = second_half - first_half (positive = accelerating)
        return second_half - first_half
    
    def _calculate_trend_strength(self, count: int, velocity: float, momentum: float) -> float:
        """Calculate overall trend strength score."""
        # Normalize factors
        count_score = min(count / 10, 1.0)  # Cap at 10 mentions
        velocity_score = min(velocity / 2, 1.0)  # Cap at 2 mentions/hour
        momentum_score = min(max(momentum / 5, -1.0), 1.0)  # Range -1 to 1
        
        # Weighted combination
        strength = (count_score * 0.4 + velocity_score * 0.4 + momentum_score * 0.2)
        return round(strength, 3)
    
    def analyze_sentiment_shifts(self, hours_back: int = 24) -> Dict[str, Any]:
        """Analyze sentiment shifts over time."""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        # Filter recent sentiment data
        recent_sentiments = {
            sentiment: [s for s in data if datetime.fromisoformat(s["timestamp"]) > cutoff_time]
            for sentiment, data in self.sentiment_history.items()
        }
        
        # Calculate sentiment trends
        sentiment_trends = {}
        for sentiment, data in recent_sentiments.items():
            if not data:
                continue
            
            # Count mentions over time
            hourly_counts = defaultdict(int)
            for entry in data:
                hour = datetime.fromisoformat(entry["timestamp"]).replace(minute=0, second=0, microsecond=0)
                hourly_counts[hour] += 1
            
            # Calculate trend direction
            hours = sorted(hourly_counts.keys())
            if len(hours) >= 2:
                first_half = sum(hourly_counts[h] for h in hours[:len(hours)//2])
                second_half = sum(hourly_counts[h] for h in hours[len(hours)//2:])
                trend_direction = "increasing" if second_half > first_half else "decreasing" if second_half < first_half else "stable"
            else:
                trend_direction = "stable"
            
            sentiment_trends[sentiment] = {
                "total_mentions": len(data),
                "trend_direction": trend_direction,
                "hourly_distribution": dict(hourly_counts),
                "themes": list(set(t for entry in data for t in entry.get("themes", [])))
            }
        
        return sentiment_trends
    
    def forecast_trends(self, hours_ahead: int = 24) -> List[Dict[str, Any]]:
        """Forecast potential trends based on current patterns."""
        current_trends = self.detect_trending_topics()
        forecasts = []
        
        for trend in current_trends:
            theme = trend["theme"]
            current_velocity = trend["velocity"]
            momentum = trend["momentum"]
            
            # Simple linear projection
            projected_mentions = trend["mention_count"] + (current_velocity * hours_ahead)
            
            # Adjust based on momentum
            if momentum > 0:
                projected_mentions *= (1 + momentum * 0.1)  # Accelerating
            elif momentum < 0:
                projected_mentions *= (1 + momentum * 0.05)  # Decelerating
            
            # Calculate confidence based on data quality
            confidence = min(trend["trend_strength"] * 0.8 + 0.2, 1.0)
            
            # Determine forecast category
            if projected_mentions > trend["mention_count"] * 1.5:
                category = "strong_growth"
            elif projected_mentions > trend["mention_count"] * 1.1:
                category = "moderate_growth"
            elif projected_mentions < trend["mention_count"] * 0.9:
                category = "declining"
            else:
                category = "stable"
            
            forecasts.append({
                "theme": theme,
                "current_mentions": trend["mention_count"],
                "projected_mentions": round(projected_mentions, 1),
                "growth_rate": round((projected_mentions / trend["mention_count"] - 1) * 100, 1),
                "forecast_category": category,
                "confidence": round(confidence, 3),
                "factors": {
                    "current_velocity": current_velocity,
                    "momentum": momentum,
                    "sentiment": trend["dominant_sentiment"],
                    "virality": trend["average_virality"]
                }
            })
        
        # Sort by projected growth
        forecasts.sort(key=lambda x: x["growth_rate"], reverse=True)
        return forecasts
    
    def generate_trend_report(self) -> Dict[str, Any]:
        """Generate comprehensive trend analysis report."""
        current_trends = self.detect_trending_topics()
        sentiment_shifts = self.analyze_sentiment_shifts()
        forecasts = self.forecast_trends()
        
        # Calculate overall platform activity
        platform_activity = defaultdict(int)
        for trend in self.trends_history[-100:]:  # Last 100 entries
            platform_activity[trend["platform"]] += 1
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "analysis_period": "24_hours",
            "summary": {
                "total_content_analyzed": len(self.trends_history),
                "active_platforms": dict(platform_activity),
                "trending_topics_count": len(current_trends),
                "sentiment_distribution": {k: len(v) for k, v in sentiment_shifts.items()}
            },
            "top_trending_topics": current_trends[:10],
            "sentiment_analysis": sentiment_shifts,
            "trend_forecasts": forecasts[:10],
            "insights": self._generate_insights(current_trends, sentiment_shifts, forecasts)
        }
        
        return report
    
    def _generate_insights(self, trends: List[Dict], sentiments: Dict, forecasts: List[Dict]) -> List[str]:
        """Generate actionable insights from trend data."""
        insights = []
        
        # Top trending insight
        if trends:
            top_trend = trends[0]
            insights.append(f"'{top_trend['theme']}' is the most trending topic with {top_trend['mention_count']} mentions and {top_trend['velocity']} mentions/hour")
        
        # Sentiment shift insights
        for sentiment, data in sentiments.items():
            if data["trend_direction"] == "increasing":
                insights.append(f"{sentiment.capitalize()} sentiment is on the rise, indicating potential mood shifts")
        
        # Forecast insights
        strong_growth = [f for f in forecasts if f["forecast_category"] == "strong_growth"]
        if strong_growth:
            insights.append(f"{len(strong_growth)} topics showing strong growth potential for the next 24 hours")
        
        # Platform insights
        if len(trends) > 5:
            insights.append("High topic diversity suggests active social media engagement across multiple subjects")
        
        return insights[:5]  # Top 5 insights


def analyze_content_trends(content_data: Dict[str, Any], analyzer: TrendAnalyzer) -> Dict[str, Any]:
    """Analyze trends for a single piece of content."""
    # Add content to trend analyzer
    analyzer.add_content(content_data)
    
    # Get current trend context
    current_trends = analyzer.detect_trending_topics()
    
    # Find relevant trends for this content
    content_themes = [t["theme"] for t in content_data.get("theme_categorization", [])]
    relevant_trends = [
        trend for trend in current_trends 
        if trend["theme"] in content_themes
    ]
    
    # Calculate trend relevance score
    trend_relevance = 0
    if relevant_trends:
        max_trend_strength = max(t["trend_strength"] for t in relevant_trends)
        trend_relevance = min(max_trend_strength, 1.0)
    
    return {
        "trend_relevance_score": round(trend_relevance, 3),
        "relevant_trends": relevant_trends,
        "trending_topics_count": len(current_trends),
        "content_added_to_trends": True
    }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python trend_analyzer.py <command> [args...]", file=sys.stderr)
        print("Commands: trends, sentiment, forecast, report", file=sys.stderr)
        raise SystemExit(1)
    
    command = sys.argv[1]
    analyzer = TrendAnalyzer()
    
    if command == "trends":
        trends = analyzer.detect_trending_topics()
        print(json.dumps(trends, ensure_ascii=False, indent=2))
    elif command == "sentiment":
        shifts = analyzer.analyze_sentiment_shifts()
        print(json.dumps(shifts, ensure_ascii=False, indent=2))
    elif command == "forecast":
        forecasts = analyzer.forecast_trends()
        print(json.dumps(forecasts, ensure_ascii=False, indent=2))
    elif command == "report":
        report = analyzer.generate_trend_report()
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"Unknown command: {command}", file=sys.stderr)
        raise SystemExit(1)
