import re
from typing import List, Optional
import logging
from textblob import TextBlob

logger = logging.getLogger(__name__)

class TextPreprocessor:
    """Preprocess text data (news, reports)"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean text data"""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^a-z0-9\s.,!?]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    @staticmethod
    def extract_keywords(text: str, top_n: int = 10) -> List[str]:
        """Extract important keywords"""
        
        # Simple keyword extraction (can be enhanced with TF-IDF)
        words = text.lower().split()
        
        # Remove common words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
            'to', 'for', 'of', 'with', 'is', 'was', 'are', 'were'
        }
        
        keywords = [w for w in words if w not in stop_words and len(w) > 3]
        
        # Count frequency
        from collections import Counter
        word_freq = Counter(keywords)
        
        return [word for word, _ in word_freq.most_common(top_n)]
    
    @staticmethod
    def analyze_sentiment(text: str) -> Dict:
        """Analyze sentiment of text"""
        
        try:
            blob = TextBlob(text)
            
            return {
                'polarity': blob.sentiment.polarity,  # -1 to 1
                'subjectivity': blob.sentiment.subjectivity,  # 0 to 1
                'sentiment': 'positive' if blob.sentiment.polarity > 0.1 
                            else 'negative' if blob.sentiment.polarity < -0.1 
                            else 'neutral'
            }
        except Exception as e:
            logger.error(f"Sentiment analysis error: {str(e)}")
            return {'polarity': 0, 'subjectivity': 0.5, 'sentiment': 'neutral'}
    
    @staticmethod
    def extract_financial_entities(text: str) -> Dict:
        """Extract financial entities (numbers, percentages, etc.)"""
        
        entities = {
            'percentages': re.findall(r'\d+\.?\d*\s*%', text),
            'currency_amounts': re.findall(r'(?:Rs\.?|₹|INR)\s*\d+(?:,\d+)*(?:\.\d+)?(?:\s*(?:crore|lakh|million|billion))?', text, re.IGNORECASE),
            'numbers': re.findall(r'\d+\.?\d*', text)
        }
        
        return entities
    