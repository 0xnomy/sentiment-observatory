#!/usr/bin/env python3
"""
Sentiment Observatory - Advanced Content Analysis Platform
Main entry point for content analysis across multiple platforms
"""

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from audio_transcriber import transcribe_youtube_audio
from audio_transcriber import transcribe_tiktok_audio
from scraper import scrape_content, scrape_content_subprocess
from advanced_analyzer import analyze_content_advanced
from virality_agent import score_virality
from trend_analyzer import TrendAnalyzer, analyze_content_trends


class ContentAnalysisPipeline:    
    def __init__(self, enable_advanced: bool = True):
        load_dotenv()
        
        self.enable_advanced = enable_advanced
        
        if self.enable_advanced:
            
            # Configuration from environment
            self.enable_trends = False  # Temporarily disabled
            self.enable_vectors = False  # Temporarily disabled
            self.enable_cross_platform = os.getenv("ENABLE_CROSS_PLATFORM", "true").lower() == "true"
        
        # Output directories
        self.reports_dir = Path("reports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Data directories
        self.data_dir = Path("data")
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def detect_platform(self, url: str) -> str:
        url = url.strip().lower()
        
        if "youtube.com" in url or "youtu.be" in url:
            return "youtube"
        elif "tiktok.com" in url:
            return "tiktok"
        elif "linkedin.com" in url:
            return "linkedin"
        elif "reddit.com" in url:
            return "reddit"
        elif "twitter.com" in url or "x.com" in url:
            return "x"
        else:
            return "unknown"
    
    def process_youtube_content(self, url: str) -> Dict[str, Any]:
        print("Processing YouTube content...")
        
        # Transcribe audio
        transcript = transcribe_youtube_audio(url, output_dir="output")
        
        # Create content data structure
        content_data = {
            "platform": "youtube",
            "url": url,
            "text": transcript,
            "body": transcript,
            "summary": transcript[:500],
            "author": "YouTube Creator",
            "timestamp": datetime.now().isoformat(),
            "images": [],
            "counts": {},
            "slug": f"youtube_{datetime.now().timestamp()}"
        }
        
        if self.enable_advanced:
            return self._run_advanced_analysis(content_data)
        else:
            return self._run_basic_analysis(content_data)
    
    def process_social_content(self, url: str) -> Dict[str, Any]:
        """Process social media content through the pipeline."""
        platform = self.detect_platform(url)
        print(f"Processing {platform} content...")
        
        # Scrape real content so images are available for multimodal analysis
        # Use subprocess scraper in server context to avoid Playwright sync/async conflicts
        try:
            if os.getenv("SERVER_CONTEXT", "false").lower() in {"1", "true", "yes"}:
                scraped_data = scrape_content_subprocess(url)
            else:
                scraped_data = scrape_content(url)
        except Exception as e:
            print(f"Warning: Scraping failed: {e}")
            scraped_data = {"text": "", "body": "", "author": "", "headline": "", "timestamp": "", "images": [], "counts": {}, "slug": f"{platform}_{datetime.now().timestamp()}"}
        
        # Build content payload for analyzers (includes images)
        content_data = {
            "platform": platform,
            "url": url,
            "text": scraped_data.get("text", "") or scraped_data.get("body", ""),
            "body": scraped_data.get("body", ""),
            "summary": (scraped_data.get("text", "") or scraped_data.get("body", ""))[:500],
            "author": scraped_data.get("author", ""),
            "headline": scraped_data.get("headline", ""),
            "timestamp": scraped_data.get("timestamp", ""),
            "images": [Path(p).as_posix() for p in scraped_data.get("images", [])],  # POSIX for portability
            "counts": scraped_data.get("counts", {}),
            "slug": scraped_data.get("slug", f"{platform}_{datetime.now().timestamp()}")
        }
        
        if self.enable_advanced:
            return self._run_advanced_analysis(content_data)
        else:
            return self._run_basic_analysis(content_data)
    
    def _run_basic_analysis(self, content_data: Dict[str, Any]) -> Dict[str, Any]:
        print("Running basic analysis...")
        
        try:
            # Basic content analysis
            analysis = analyze_content_advanced(
                content_data["text"],
                image_path=content_data["images"][0] if content_data["images"] else None,
                platform=content_data["platform"]
            )
        except Exception as e:
            print(f"Warning: Content analysis failed: {e}")
            # Create a simple fallback analysis
            text = content_data.get("text", "")
            analysis = {
                "summary": text[:200] + "..." if len(text) > 200 else text,
                "sentiment": {"label": "neutral", "reason": "Analysis failed"},
                "virality_score": 5,
                "report": "Content analysis completed with fallback methods",
                "themes": [],
                "content_type": "unknown",
                "target_audience": "general",
                "engagement_potential": "medium",
                "visual_analysis": "Visual analysis not available",
                "error": str(e)
            }
        
        try:
            # Virality scoring
            virality = score_virality(content_data["text"][:1500])
        except Exception as e:
            print(f"Warning: Virality scoring failed: {e}")
            # Create a simple fallback virality score
            virality = {
                "virality_score": 5,
                "reason": "Fallback score due to analysis failure",
                "error": str(e)
            }
        
        return {
            "platform": content_data["platform"],
            "url": content_data["url"],
            "scraped": content_data,
            "analysis": analysis,
            "virality_agent": virality,
            "timestamp": datetime.now().isoformat()
        }
    
    def _run_advanced_analysis(self, content_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run advanced content analysis with all engines."""
        print("Running advanced analysis...")
        
        try:
            # Advanced content analysis
            analysis = analyze_content_advanced(
                content_data["text"],
                image_path=content_data["images"][0] if content_data["images"] else None,
                platform=content_data["platform"]
            )
            
            # Virality scoring
            virality = score_virality(content_data["text"][:1500])
            
            # Initialize result
            result = {
                "platform": content_data["platform"],
                "url": content_data["url"],
                "scraped": content_data,
                "analysis": analysis,
                "virality_agent": virality,
                "timestamp": datetime.now().isoformat()
            }
            
            # Add advanced analysis if enabled (trend/vector disabled)
            return result
            
        except Exception as e:
            print(f"Warning: Advanced analysis failed: {e}")
            # Fallback to basic analysis
            return self._run_basic_analysis(content_data)
    
    def generate_human_readable_report(self, report: Dict[str, Any]) -> str:
        """Generate a comprehensive human-readable version of the analysis report."""
        lines = []
        lines.append("=" * 80)
        lines.append("🎯 COMPREHENSIVE CONTENT ANALYSIS REPORT")
        lines.append("=" * 80)
        
        # Header information
        lines.append(f"📱 Platform: {report.get('platform', 'Unknown').upper()}")
        lines.append(f"🔗 Source URL: {report.get('url', 'N/A')}")
        lines.append(f"⏰ Analysis Timestamp: {report.get('timestamp', 'N/A')}")
        lines.append("")
        
        # Executive Summary Section
        lines.append("🚀 EXECUTIVE SUMMARY")
        lines.append("=" * 50)
        scraped = report.get("scraped", {})
        analysis = report.get("analysis", {})
        
        # Create a comprehensive summary
        summary_parts = []
        if scraped.get("author"):
            summary_parts.append(f"Content by {scraped['author']}")
        if scraped.get("headline"):
            summary_parts.append(f"Headline: '{scraped['headline']}'")
        if analysis.get("summary"):
            summary_parts.append(f"AI Summary: {analysis['summary']}")
        
        if summary_parts:
            lines.append(" • ".join(summary_parts))
        else:
            lines.append("Content analysis completed with available data")
        lines.append("")
        
        # Content Overview Section
        lines.append("📋 CONTENT OVERVIEW")
        lines.append("-" * 40)
        
        # Text content analysis
        if scraped.get("text"):
            text_content = scraped["text"]
            text_preview = text_content[:200] + "..." if len(text_content) > 200 else text_content
            lines.append(f"📝 Text Content Preview:")
            lines.append(f"   {text_preview}")
            lines.append(f"   Total Length: {len(text_content)} characters")
            lines.append("")
        
        # Image content analysis
        if scraped.get("images"):
            lines.append(f"🖼️ Visual Content:")
            lines.append(f"   • {len(scraped['images'])} image(s) detected and downloaded")
            for i, img_path in enumerate(scraped['images'][:3], 1):  # Show first 3 images
                img_name = Path(img_path).name
                lines.append(f"   • Image {i}: {img_name}")
            if len(scraped['images']) > 3:
                lines.append(f"   • ... and {len(scraped['images']) - 3} more images")
            lines.append("")
        
        # Engagement metrics
        if scraped.get("counts"):
            counts = scraped["counts"]
            engagement_metrics = []
            if counts.get("upvotes") or counts.get("likes"):
                engagement_metrics.append(f"👍 {counts.get('upvotes') or counts.get('likes')} upvotes/likes")
            if counts.get("comments"):
                engagement_metrics.append(f"💬 {counts['comments']} comments")
            if counts.get("reposts") or counts.get("retweets"):
                engagement_metrics.append(f"🔄 {counts.get('reposts') or counts.get('retweets')} reposts/retweets")
            if counts.get("replies"):
                engagement_metrics.append(f"↩️ {counts['replies']} replies")
            
            if engagement_metrics:
                lines.append("📊 Engagement Metrics:")
                lines.append("   " + " | ".join(engagement_metrics))
                lines.append("")
        
        # AI Analysis Section
        if analysis:
            lines.append("🤖 AI-POWERED ANALYSIS")
            lines.append("=" * 50)
            
            # Sentiment Analysis
            sentiment = analysis.get("sentiment", {})
            if isinstance(sentiment, dict):
                sentiment_label = sentiment.get('label', 'N/A')
                sentiment_reason = sentiment.get('reason', '')
                lines.append(f"💭 Sentiment Analysis:")
                lines.append(f"   • Overall Sentiment: {sentiment_label.upper()}")
                if sentiment_reason:
                    lines.append(f"   • Reasoning: {sentiment_reason}")
                lines.append("")
            else:
                lines.append(f"💭 Sentiment: {sentiment}")
                lines.append("")
            
            # Content Type and Audience
            if analysis.get("content_type"):
                lines.append(f"🏷️ Content Classification:")
                lines.append(f"   • Type: {analysis['content_type'].title()}")
                if analysis.get("target_audience"):
                    lines.append(f"   • Target Audience: {analysis['target_audience'].replace('_', ' ').title()}")
                if analysis.get("engagement_potential"):
                    lines.append(f"   • Engagement Potential: {analysis['engagement_potential'].title()}")
                lines.append("")
            
            # Theme Analysis
            if analysis.get("themes"):
                lines.append(f"🎨 Theme Analysis:")
                for theme in analysis["themes"][:5]:  # Show top 5 themes
                    if isinstance(theme, dict):
                        theme_name = theme.get("theme", "Unknown")
                        confidence = theme.get("confidence", 0)
                        lines.append(f"   • {theme_name} (Confidence: {confidence:.1f})")
                    else:
                        lines.append(f"   • {theme}")
                lines.append("")
            
            # Virality Score
            if analysis.get("virality_score"):
                lines.append(f"📈 Virality Assessment:")
                lines.append(f"   • Score: {analysis['virality_score']}/10")
                if analysis.get("report"):
                    lines.append(f"   • AI Report: {analysis['report']}")
                lines.append("")
        
        # Virality Agent Section
        va = report.get("virality_agent", {})
        if va:
            lines.append("🚀 ADVANCED VIRALITY ASSESSMENT")
            lines.append("-" * 50)
            lines.append(f"🎯 LangChain Virality Score: {va.get('virality_score', 'N/A')}/10")
            if va.get('reason'):
                lines.append(f"💡 Reasoning: {va['reason']}")
            lines.append("")
        
        # Advanced Analytics Sections
        if report.get("trend_analysis"):
            lines.append("📈 TREND ANALYSIS INSIGHTS")
            lines.append("-" * 50)
            trend = report["trend_analysis"]
            if trend.get("trend_strength"):
                lines.append(f"📊 Trend Strength: {trend['trend_strength']}")
            if trend.get("momentum"):
                lines.append(f"🚀 Momentum: {trend['momentum']}")
            if trend.get("velocity"):
                lines.append(f"⚡ Velocity: {trend['velocity']}")
            lines.append("")
        
        # Content Context and Insights
        lines.append("💡 CONTENT INSIGHTS & RECOMMENDATIONS")
        lines.append("=" * 50)
        
        # Generate insights based on analysis
        insights = []
        # Safely coerce virality score
        def _to_float(value, default=0.0):
            try:
                return float(value)
            except Exception:
                return float(default)
        def _to_int(value, default=0):
            try:
                return int(str(value).replace(',', '').strip())
            except Exception:
                return int(default)
        
        if analysis.get("sentiment", {}).get("label") == "positive":
            insights.append("✅ Content has positive sentiment, likely to generate engagement")
        elif analysis.get("sentiment", {}).get("label") == "negative":
            insights.append("⚠️ Content has negative sentiment, may require careful monitoring")
        
        if scraped.get("images"):
            insights.append("🖼️ Visual content detected - enhances engagement potential")
        
        virality_score_num = _to_float(analysis.get("virality_score", 0))
        if virality_score_num >= 7:
            insights.append("🔥 High virality potential - consider amplification strategies")
        elif virality_score_num <= 3:
            insights.append("📉 Low virality potential - may need content optimization")
        
        comments_count = _to_int(scraped.get("counts", {}).get("comments", 0))
        if comments_count > 10:
            insights.append("💬 High comment activity indicates strong community engagement")
        
        if insights:
            for insight in insights:
                lines.append(f"   {insight}")
        else:
            lines.append("   Content analysis completed with standard metrics")
        lines.append("")
        
        # Technical Details
        lines.append("🔧 TECHNICAL DETAILS")
        lines.append("-" * 40)
        lines.append(f"   • Analysis Engine: Groq LLaMA-4 Maverick")
        lines.append(f"   • Multimodal Analysis: {'Enabled' if scraped.get('images') else 'Text-only'}")
        lines.append(f"   • Advanced Features: {'Enabled' if report.get('trend_analysis') or report.get('vector_analysis') else 'Basic'}")
        lines.append(f"   • Data Persistence: {'Enabled' if report.get('trend_analysis') else 'Disabled'}")
        lines.append("")
        
        lines.append("=" * 80)
        lines.append("📋 Report generated by Sentiment Observatory")
        lines.append("=" * 80)
        return "\n".join(lines)
    
    def save_report(self, report: Dict[str, Any]) -> str:
        """Save the analysis report to file."""
        platform = report.get("platform", "unknown")
        slug = report.get("scraped", {}).get("slug", f"report_{datetime.now().timestamp()}")
        # Append timestamp to avoid collisions
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        unique_slug = f"{slug}_{ts}"
        
        # Create platform-specific directory
        platform_dir = self.reports_dir / platform
        platform_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON report
        json_path = platform_dir / f"{unique_slug}.json"
        json_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8"
        )
        
        # Save human-readable report
        txt_path = platform_dir / f"{unique_slug}.txt"
        txt_path.write_text(
            self.generate_human_readable_report(report),
            encoding="utf-8"
        )
        
        print(f"Reports saved to:")
        print(f"  JSON: {json_path}")
        print(f"  Text: {txt_path}")
        
        return str(json_path)
    
    def run_pipeline(self, url: str) -> Dict[str, Any]:
        """Main pipeline execution."""
        platform = self.detect_platform(url)
        
        if platform == "youtube":
            return self.process_youtube_content(url)
        elif platform == "tiktok":
            # Process TikTok as audio transcription pipeline
            print("Processing TikTok content...")
            transcript = transcribe_tiktok_audio(url, output_dir="output")
            content_data = {
                "platform": "tiktok",
                "url": url,
                "text": transcript,
                "body": transcript,
                "summary": transcript[:500],
                "author": "TikTok Creator",
                "timestamp": datetime.now().isoformat(),
                "images": [],
                "counts": {},
                "slug": f"tiktok_{datetime.now().timestamp()}"
            }
            return self._run_advanced_analysis(content_data) if self.enable_advanced else self._run_basic_analysis(content_data)
        elif platform in {"linkedin", "reddit", "x"}:
            return self.process_social_content(url)
        else:
            raise ValueError(f"Unsupported platform: {platform}. Supported: YouTube, TikTok, LinkedIn, Reddit, X/Twitter")
    
    def generate_system_report(self) -> Dict[str, Any]:
        """Generate a system-wide analysis report."""
        if not self.enable_advanced:
            return {"message": "Advanced features not enabled"}
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "system_status": "operational"
        }
        
        if self.enable_trends:
            try:
                trend_report = self.trend_analyzer.generate_trend_report()
                report["trends"] = trend_report
            except Exception as e:
                report["trends_error"] = str(e)
        
        # Vector analysis temporarily disabled
        # if self.enable_vectors:
        #     try:
        #         vector_report = self.vector_engine.generate_vector_report()
        #         report["vectors"] = vector_report
        #     except Exception as e:
        #         report["vectors_error"] = str(e)
        
        return report


def main():
    """Main entry point."""
    load_dotenv()
    
    if len(sys.argv) < 2:
        print("Sentiment Observatory - Advanced Content Analysis Platform")
        print("=" * 60)
        print("Usage:")
        print("  python app.py <url>                    # Basic analysis")
        print("  python app.py <url> --advanced         # Advanced analysis (default)")
        print("  python app.py <url> --basic            # Basic analysis only")
        print("  python app.py --system                 # Generate system report")
        print("")
        print("Examples:")
        print("  python app.py https://youtube.com/watch?v=...")
        print("  python app.py https://linkedin.com/posts/...")
        print("  python app.py https://reddit.com/r/...")
        print("  python app.py https://twitter.com/...")
        print("")
        print("Environment Variables:")
        print("  GROQ_API_KEY           # Required for AI analysis")
        print("  ENABLE_TRENDS          # Enable trend analysis (default: true)")
        print("  ENABLE_VECTORS         # Enable vector analysis (default: false - temporarily disabled)")
        print("  ENABLE_CROSS_PLATFORM  # Enable cross-platform correlation (default: true)")
        print("  HEADLESS               # Browser headless mode (default: false)")
        print("  SESSION_DIR            # Playwright session directory (default: .playwright/session)")
        print("  SCRAPER_TIMEOUT_MS     # Scraper timeout in ms (default: 20000)")
        sys.exit(1)
    
    # Parse arguments
    args = sys.argv[1:]
    url = None
    enable_advanced = True
    
    for arg in args:
        if arg == "--advanced":
            enable_advanced = True
        elif arg == "--basic":
            enable_advanced = False
        elif arg == "--system":
            # Generate system report
            pipeline = ContentAnalysisPipeline(enable_advanced=True)
            report = pipeline.generate_system_report()
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return
        elif not arg.startswith("--"):
            url = arg
    
    if not url:
        print("Error: No URL provided", file=sys.stderr)
        sys.exit(1)
    
    try:
        # Initialize pipeline
        pipeline = ContentAnalysisPipeline(enable_advanced=enable_advanced)
        
        # Run analysis
        report = pipeline.run_pipeline(url)
        
        # Save report
        pipeline.save_report(report)
        
        # Print human-readable version
        print("\n" + pipeline.generate_human_readable_report(report))
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
