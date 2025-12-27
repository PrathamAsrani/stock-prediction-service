from typing import List, Dict, Optional
from app.knowledge_base.vector_store import VectorStore
from app.config.settings import settings
import logging

logger = logging.getLogger(__name__)

class RAGService:
    """Retrieval-Augmented Generation service for trading insights"""
    
    def __init__(self):
        self.vector_store = VectorStore(
            persist_directory=settings.KNOWLEDGE_BASE_PATH,
            embedding_model=settings.EMBEDDING_MODEL
        )
    
    def get_trading_insights(
        self,
        symbol: str,
        query: str = "swing trading strategy",
        include_patterns: bool = True,
        include_fundamentals: bool = True
    ) -> Dict[str, any]:
        """Get comprehensive trading insights using RAG"""
        
        insights = {
            'symbol': symbol,
            'query': query,
            'book_insights': [],
            'company_insights': [],
            'pattern_insights': [],
            'recommendations': []
        }
        
        # 1. Get insights from trading books
        book_query = f"swing trading strategy for stocks like {symbol}"
        book_results = self.vector_store.search(
            query=book_query,
            collection_type='books',
            n_results=5
        )
        
        for result in book_results:
            insights['book_insights'].append({
                'content': result['document'],
                'source': result['metadata'].get('filename', 'Unknown'),
                'category': result['metadata'].get('category', 'general')
            })
        
        # 2. Get company-specific insights from reports
        if include_fundamentals:
            company_results = self.vector_store.search(
                query=f"financial performance analysis {symbol}",
                collection_type='reports',
                n_results=5,
                filter_metadata={'company': symbol} if symbol else None
            )
            
            for result in company_results:
                insights['company_insights'].append({
                    'content': result['document'],
                    'year': result['metadata'].get('year', 'N/A'),
                    'type': result['metadata'].get('type', 'report'),
                    'metrics': result['metadata'].get('metrics', {})
                })
        
        # 3. Get historical pattern insights
        if include_patterns:
            pattern_results = self.vector_store.search(
                query=f"chart patterns {symbol}",
                collection_type='patterns',
                n_results=10,
                filter_metadata={'symbol': symbol} if symbol else None
            )
            
            for result in pattern_results:
                insights['pattern_insights'].append({
                    'content': result['document'],
                    'pattern_name': result['metadata'].get('pattern_name', 'Unknown'),
                    'outcome': result['metadata'].get('outcome', 'unknown'),
                    'price_change': result['metadata'].get('price_change', None)
                })
        
        # 4. Generate recommendations based on retrieved context
        recommendations = self._generate_recommendations(insights)
        insights['recommendations'] = recommendations
        
        return insights
    
    def _generate_recommendations(self, insights: Dict) -> List[str]:
        """Generate actionable recommendations from insights"""
        recommendations = []
        
        # Analyze patterns
        if insights['pattern_insights']:
            bullish_patterns = sum(
                1 for p in insights['pattern_insights'] 
                if p.get('outcome') == 'bullish'
            )
            bearish_patterns = sum(
                1 for p in insights['pattern_insights'] 
                if p.get('outcome') == 'bearish'
            )
            
            if bullish_patterns > bearish_patterns:
                recommendations.append(
                    f"Historical pattern analysis shows {bullish_patterns} bullish patterns vs {bearish_patterns} bearish patterns, suggesting positive momentum potential"
                )
            elif bearish_patterns > bullish_patterns:
                recommendations.append(
                    f"Historical pattern analysis shows {bearish_patterns} bearish patterns vs {bullish_patterns} bullish patterns, exercise caution"
                )
        
        # Analyze fundamentals from reports
        if insights['company_insights']:
            recent_reports = [
                r for r in insights['company_insights']
                if r.get('metrics')
            ]
            
            if recent_reports:
                recommendations.append(
                    f"Company fundamentals available from {len(recent_reports)} reports. Review financial metrics before trading."
                )
        
        # Book-based insights
        if insights['book_insights']:
            technical_books = sum(
                1 for b in insights['book_insights']
                if b.get('category') == 'technical_analysis'
            )
            
            if technical_books > 0:
                recommendations.append(
                    f"Found {technical_books} relevant technical analysis strategies applicable to this stock"
                )
        
        return recommendations
    
    def get_pattern_context(self, pattern_name: str, outcome: str = None) -> str:
        """Get detailed context about a specific pattern"""
        
        query = f"{pattern_name} pattern trading strategy"
        
        filter_metadata = {'pattern_name': pattern_name}
        if outcome:
            filter_metadata['outcome'] = outcome
        
        # Search in patterns collection
        pattern_results = self.vector_store.search(
            query=query,
            collection_type='patterns',
            n_results=5,
            filter_metadata=filter_metadata
        )
        
        # Search in books for pattern strategy
        book_results = self.vector_store.search(
            query=f"{pattern_name} trading",
            collection_type='books',
            n_results=3
        )
        
        # Compile context
        context = f"### {pattern_name.replace('_', ' ').title()} Pattern\n\n"
        
        if pattern_results:
            context += "**Historical Occurrences:**\n"
            for result in pattern_results:
                context += f"- {result['document']}\n"
        
        if book_results:
            context += "\n**Trading Strategies:**\n"
            for result in book_results:
                context += f"- From {result['metadata'].get('filename', 'Unknown')}: {result['document'][:200]}...\n"
        
        return context
    
    def get_swing_trading_strategy(self, symbol: str) -> Dict:
        """Get comprehensive swing trading strategy"""
        
        strategy = {
            'symbol': symbol,
            'entry_criteria': [],
            'exit_criteria': [],
            'risk_management': [],
            'supporting_evidence': []
        }
        
        # Search for swing trading strategies in books
        strategy_results = self.vector_store.search(
            query="swing trading entry exit strategy risk management",
            collection_type='books',
            n_results=10
        )
        
        # Categorize insights
        for result in strategy_results:
            content = result['document'].lower()
            
            if any(word in content for word in ['entry', 'buy signal', 'enter position']):
                strategy['entry_criteria'].append(result['document'])
            
            if any(word in content for word in ['exit', 'sell signal', 'take profit', 'stop loss']):
                strategy['exit_criteria'].append(result['document'])
            
            if any(word in content for word in ['risk', 'position size', 'stop loss', 'risk management']):
                strategy['risk_management'].append(result['document'])
        
        # Get company-specific supporting evidence
        company_evidence = self.vector_store.search(
            query=f"{symbol} stock analysis performance",
            collection_type='reports',
            n_results=3,
            filter_metadata={'company': symbol}
        )
        
        strategy['supporting_evidence'] = [r['document'] for r in company_evidence]
        
        return strategy
    