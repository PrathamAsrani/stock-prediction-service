import pandas as pd
import numpy as np
from typing import List, Dict
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import logging

logger = logging.getLogger(__name__)

class SentimentFeaturesExtractor:
    """Extract sentiment features from news and text data"""
    
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
    
    def extract_features(self, texts: List[str]) -> Dict:
        """
        Extract sentiment features from list of texts (news articles, reports, etc.)
        
        Args:
            texts: List of text strings
            
        Returns:
            Dictionary of sentiment features
        """
        if not texts:
            return self._get_neutral_features()
        
        features = {}
        
        # Calculate sentiment for each text
        sentiments = []
        for text in texts:
            sentiment = self.analyze_single_text(text)
            sentiments.append(sentiment)
        
        # Aggregate sentiments
        df = pd.DataFrame(sentiments)
        
        # === OVERALL SENTIMENT ===
        features['sentiment_mean'] = df['compound'].mean()
        features['sentiment_std'] = df['compound'].std()
        features['sentiment_min'] = df['compound'].min()
        features['sentiment_max'] = df['compound'].max()
        
        # === SENTIMENT CATEGORIES ===
        features['positive_ratio'] = (df['compound'] > 0.05).sum() / len(df)
        features['negative_ratio'] = (df['compound'] < -0.05).sum() / len(df)
        features['neutral_ratio'] = ((df['compound'] >= -0.05) & (df['compound'] <= 0.05)).sum() / len(df)
        
        # === SENTIMENT STRENGTH ===
        features['strong_positive_ratio'] = (df['compound'] > 0.5).sum() / len(df)
        features['strong_negative_ratio'] = (df['compound'] < -0.5).sum() / len(df)
        
        # === POLARITY AND SUBJECTIVITY ===
        features['polarity_mean'] = df['polarity'].mean()
        features['subjectivity_mean'] = df['subjectivity'].mean()
        
        # === RECENT SENTIMENT (if timestamps available) ===
        # Weight recent news more heavily
        if len(df) > 5:
            recent = df.tail(5)
            features['recent_sentiment'] = recent['compound'].mean()
        else:
            features['recent_sentiment'] = features['sentiment_mean']
        
        # === SENTIMENT TREND ===
        # Is sentiment improving or declining?
        if len(df) >= 10:
            first_half = df.head(len(df)//2)['compound'].mean()
            second_half = df.tail(len(df)//2)['compound'].mean()
            features['sentiment_trend'] = second_half - first_half
        else:
            features['sentiment_trend'] = 0.0
        
        # === COMPOSITE SENTIMENT SCORE (0-100) ===
        features['sentiment_score'] = self._calculate_sentiment_score(features)
        
        logger.info(f"Extracted sentiment features from {len(texts)} texts")
        
        return features
    
    def analyze_single_text(self, text: str) -> Dict:
        """Analyze sentiment of a single text"""
        
        # VADER sentiment (better for short texts, social media)
        vader_scores = self.vader.polarity_scores(text)
        
        # TextBlob sentiment
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
        except:
            polarity = 0
            subjectivity = 0.5
        
        return {
            'compound': vader_scores['compound'],
            'positive': vader_scores['pos'],
            'negative': vader_scores['neg'],
            'neutral': vader_scores['neu'],
            'polarity': polarity,
            'subjectivity': subjectivity
        }
    
    def _calculate_sentiment_score(self, features: Dict) -> float:
        """Calculate composite sentiment score (0-100)"""
        
        # Base score from mean sentiment (-1 to 1 -> 0 to 100)
        base_score = (features['sentiment_mean'] + 1) * 50
        
        # Adjust for positive/negative ratios
        ratio_adjustment = (features['positive_ratio'] - features['negative_ratio']) * 20
        
        # Adjust for recent sentiment
        recent_adjustment = (features['recent_sentiment']) * 10
        
        # Adjust for trend
        trend_adjustment = features['sentiment_trend'] * 15
        
        # Combine
        score = base_score + ratio_adjustment + recent_adjustment + trend_adjustment
        
        # Clamp to 0-100
        return max(0, min(100, score))
    
    def _get_neutral_features(self) -> Dict:
        """Return neutral features when no text available"""
        return {
            'sentiment_mean': 0.0,
            'sentiment_std': 0.0,
            'sentiment_min': 0.0,
            'sentiment_max': 0.0,
            'positive_ratio': 0.33,
            'negative_ratio': 0.33,
            'neutral_ratio': 0.34,
            'strong_positive_ratio': 0.0,
            'strong_negative_ratio': 0.0,
            'polarity_mean': 0.0,
            'subjectivity_mean': 0.5,
            'recent_sentiment': 0.0,
            'sentiment_trend': 0.0,
            'sentiment_score': 50.0
        }
    
    @staticmethod
    def get_feature_names() -> List[str]:
        """Get list of sentiment feature names"""
        return [
            'sentiment_mean', 'sentiment_std', 'sentiment_min', 'sentiment_max',
            'positive_ratio', 'negative_ratio', 'neutral_ratio',
            'strong_positive_ratio', 'strong_negative_ratio',
            'polarity_mean', 'subjectivity_mean',
            'recent_sentiment', 'sentiment_trend',
            'sentiment_score'
        ]
    