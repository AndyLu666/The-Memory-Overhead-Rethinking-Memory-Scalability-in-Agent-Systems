from typing import Dict, Any, Optional
from utils.time_statistic import OverallTimeStatistic
from utils.cost_manager import GraphBuildingCostManager, QueryCostManager
from init.logger import logger


class FinalReportGenerator:
    """Generate comprehensive final reports for Dynamic Memory system."""
    
    def __init__(self):
        self.graph_building_stats = None
        self.runtime_stats = []
        self.evaluation_results = None
        self.overall_stats = None
    
    def set_graph_building_stats(self, time_stats, cost_stats):
        """Set graph building statistics."""
        self.graph_building_stats = {
            'time': time_stats,
            'cost': cost_stats
        }
    
    def add_runtime_stats(self, time_stats, cost_stats, memos_stats=None):
        """Add MemOS-style runtime statistics."""
        self.runtime_stats.append({
            'time': time_stats,
            'cost': cost_stats,
            'memos': memos_stats or {},
        })
    
    def set_evaluation_results(self, evaluation_results: Dict[str, Any]):
        """Set evaluation results."""
        self.evaluation_results = evaluation_results
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate a comprehensive final report."""
        report = {
            'system_overview': self._generate_system_overview(),
            'graph_building_summary': self._generate_graph_building_summary(),
            'memos_runtime_summary': self._generate_memos_runtime_summary(),
            'evaluation_summary': self._generate_evaluation_summary(),
            'cost_analysis': self._generate_cost_analysis(),
            'performance_analysis': self._generate_performance_analysis(),
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _generate_system_overview(self) -> Dict[str, Any]:
        """Generate system overview."""
        total_queries = len(self.runtime_stats)
        total_graph_building_cost = 0
        total_query_cost = 0
        total_graph_building_time = 0
        total_query_time = 0
        
        if self.graph_building_stats:
            total_graph_building_cost = self.graph_building_stats['cost'].get('total_cost_usd', 0)
            total_graph_building_time = self.graph_building_stats['time'].get('total_graph_building_time', 0)
        
        for runtime_stat in self.runtime_stats:
            total_query_cost += runtime_stat['cost'].get('total_cost_usd', 0)
            total_query_time += runtime_stat['time'].get('total_query_time', 0)
        
        return {
            'total_queries_processed': total_queries,
            'total_system_cost_usd': round(total_graph_building_cost + total_query_cost, 4),
            'total_system_time_seconds': round(total_graph_building_time + total_query_time, 2),
            'average_cost_per_query': round(total_query_cost / total_queries, 4) if total_queries > 0 else 0,
            'average_time_per_query': round(total_query_time / total_queries, 2) if total_queries > 0 else 0
        }
    
    def _generate_graph_building_summary(self) -> Dict[str, Any]:
        """Generate graph building summary."""
        if not self.graph_building_stats:
            return {'status': 'No graph building data available'}
        
        time_stats = self.graph_building_stats['time']
        cost_stats = self.graph_building_stats['cost']
        
        return {
            'time_breakdown': {
                'chunking_time': time_stats.get('chunking_time', 0),
                'entity_extraction_time': time_stats.get('entity_extraction_time', 0),
                'relationship_extraction_time': time_stats.get('relationship_extraction_time', 0),
                'graph_construction_time': time_stats.get('graph_construction_time', 0),
                'total_time': time_stats.get('total_graph_building_time', 0)
            },
            'cost_breakdown': {
                'chunking_tokens': cost_stats.get('chunking_tokens', 0),
                'entity_extraction_tokens': cost_stats.get('entity_extraction_tokens', 0),
                'relationship_extraction_tokens': cost_stats.get('relationship_extraction_tokens', 0),
                'graph_construction_tokens': cost_stats.get('graph_construction_tokens', 0),
                'total_tokens': cost_stats.get('total_graph_building_tokens', 0),
                'total_cost_usd': cost_stats.get('total_cost_usd', 0)
            },
            'efficiency_metrics': {
                'tokens_per_second': round(
                    cost_stats.get('total_graph_building_tokens', 0) / 
                    max(time_stats.get('total_graph_building_time', 1), 0.1), 2
                ),
                'cost_per_1000_tokens': round(
                    (cost_stats.get('total_cost_usd', 0) / 
                     max(cost_stats.get('total_graph_building_tokens', 1), 1)) * 1000, 4
                )
            }
        }
    
    def _generate_memos_runtime_summary(self) -> Dict[str, Any]:
        """Generate MemOS-style runtime summary."""
        if not self.runtime_stats:
            return {'status': 'No runtime data available'}
        
        total_queries = len(self.runtime_stats)
        total_retrieval_time = 0
        total_answer_generation_time = 0
        total_query_cost = 0
        total_context_tokens = 0
        total_response_duration_ms = 0.0
        total_search_duration_ms = 0.0
        total_total_duration_ms = 0.0

        for runtime_stat in self.runtime_stats:
            time_stats = runtime_stat['time']
            cost_stats = runtime_stat['cost']
            memos_stats = runtime_stat.get('memos') or {}
            
            total_retrieval_time += time_stats.get('retrieval_time', 0)
            total_answer_generation_time += time_stats.get('answer_generation_time', 0)
            total_query_cost += cost_stats.get('total_cost_usd', 0)
            total_context_tokens += int(memos_stats.get('context_tokens_total', 0) or 0)
            total_response_duration_ms += float(memos_stats.get('response_duration_ms_avg', 0.0) or 0.0)
            total_search_duration_ms += float(memos_stats.get('search_duration_ms_avg', 0.0) or 0.0)
            total_total_duration_ms += float(memos_stats.get('total_duration_ms_avg', 0.0) or 0.0)

        return {
            'total_queries': total_queries,
            'time_breakdown': {
                'total_retrieval_time': round(total_retrieval_time, 2),
                'total_answer_generation_time': round(total_answer_generation_time, 2),
                'total_query_time': round(total_retrieval_time + total_answer_generation_time, 2),
                'average_retrieval_time': round(total_retrieval_time / total_queries, 2) if total_queries > 0 else 0,
                'average_answer_generation_time': round(total_answer_generation_time / total_queries, 2) if total_queries > 0 else 0
            },
            'memos_breakdown': {
                'context_tokens_total': total_context_tokens,
                'context_tokens_avg': round(total_context_tokens / total_queries, 3) if total_queries > 0 else 0.0,
                'response_duration_ms_avg': round(total_response_duration_ms / total_queries, 3) if total_queries > 0 else 0.0,
                'search_duration_ms_avg': round(total_search_duration_ms / total_queries, 3) if total_queries > 0 else 0.0,
                'total_duration_ms_avg': round(total_total_duration_ms / total_queries, 3) if total_queries > 0 else 0.0,
                'total_query_cost_usd': round(total_query_cost, 4),
                'average_cost_per_query': round(total_query_cost / total_queries, 4) if total_queries > 0 else 0
            },
            'efficiency_metrics': {
                'queries_per_second': round(
                    total_queries / max(total_retrieval_time + total_answer_generation_time, 0.1), 4
                ) if total_queries > 0 else 0.0,
                'cost_per_query': round(total_query_cost / max(total_queries, 1), 4),
            }
        }
    
    def _generate_evaluation_summary(self) -> Dict[str, Any]:
        """Generate evaluation summary."""
        if not self.evaluation_results:
            return {'status': 'No evaluation results available'}
        
        return {
            'accuracy': self.evaluation_results.get('accuracy', 0),
            'correct_answers': self.evaluation_results.get('correct_answers', 0),
            'total_answers': self.evaluation_results.get('total_answers', 0),
            'answer_rate': self.evaluation_results.get('answer_rate', 0)
        }
    
    def _generate_cost_analysis(self) -> Dict[str, Any]:
        """Generate cost analysis."""
        graph_cost = self.graph_building_stats['cost'].get('total_cost_usd', 0) if self.graph_building_stats else 0
        query_cost = sum(stat['cost'].get('total_cost_usd', 0) for stat in self.runtime_stats)
        total_cost = graph_cost + query_cost
        
        return {
            'graph_building_cost': round(graph_cost, 4),
            'query_processing_cost': round(query_cost, 4),
            'total_system_cost': round(total_cost, 4),
            'cost_distribution': {
                'graph_building_percentage': round((graph_cost / max(total_cost, 0.001)) * 100, 1),
                'query_processing_percentage': round((query_cost / max(total_cost, 0.001)) * 100, 1)
            },
            'cost_efficiency': {
                'cost_per_query': round(query_cost / max(len(self.runtime_stats), 1), 4),
                'amortized_graph_cost_per_query': round(graph_cost / max(len(self.runtime_stats), 1), 4)
            }
        }
    
    def _generate_performance_analysis(self) -> Dict[str, Any]:
        """Generate performance analysis."""
        graph_time = self.graph_building_stats['time'].get('total_graph_building_time', 0) if self.graph_building_stats else 0
        query_time = sum(stat['time'].get('total_query_time', 0) for stat in self.runtime_stats)
        total_time = graph_time + query_time
        
        return {
            'graph_building_time': round(graph_time, 2),
            'query_processing_time': round(query_time, 2),
            'total_system_time': round(total_time, 2),
            'time_distribution': {
                'graph_building_percentage': round((graph_time / max(total_time, 0.001)) * 100, 1),
                'query_processing_percentage': round((query_time / max(total_time, 0.001)) * 100, 1)
            },
            'throughput_metrics': {
                'queries_per_minute': round(len(self.runtime_stats) / max(total_time / 60, 0.001), 2),
                'average_query_latency': round(query_time / max(len(self.runtime_stats), 1), 2)
            }
        }
    
    def _generate_recommendations(self) -> Dict[str, Any]:
        """Generate optimization recommendations."""
        recommendations = []
        
        # Cost optimization recommendations
        if self.graph_building_stats and self.graph_building_stats['cost'].get('total_cost_usd', 0) > 0.1:
            recommendations.append({
                'type': 'cost_optimization',
                'category': 'graph_building',
                'message': 'Consider using smaller models for entity/relationship extraction to reduce costs',
                'impact': 'high'
            })
        
        # Performance optimization recommendations
        if self.runtime_stats:
            avg_query_time = sum(stat['time'].get('total_query_time', 0) for stat in self.runtime_stats) / len(self.runtime_stats)
            if avg_query_time > 10:
                recommendations.append({
                    'type': 'performance_optimization',
                    'category': 'query_processing',
                    'message': 'Query processing time is high. Consider optimizing retrieval or using faster models',
                    'impact': 'medium'
                })
        
        # Accuracy recommendations
        if self.evaluation_results and self.evaluation_results.get('accuracy', 0) < 0.8:
            recommendations.append({
                'type': 'accuracy_improvement',
                'category': 'model_performance',
                'message': 'Accuracy is below 80%. Consider improving prompts or using more powerful models',
                'impact': 'high'
            })
        
        return {
            'total_recommendations': len(recommendations),
            'recommendations': recommendations
        }
    
    def print_final_report(self):
        """Print a formatted final report to console."""
        report = self.generate_comprehensive_report()
        
        logger.info("=" * 100)
        logger.info("🏁 FINAL COMPREHENSIVE REPORT - DYNAMIC MEMORY GRAPH RAG SYSTEM")
        logger.info("=" * 100)
        
        # System Overview
        overview = report['system_overview']
        logger.info("📊 SYSTEM OVERVIEW:")
        logger.info(f"   Total Queries Processed: {overview['total_queries_processed']}")
        logger.info(f"   Total System Cost: ${overview['total_system_cost_usd']}")
        logger.info(f"   Total System Time: {overview['total_system_time_seconds']}s")
        logger.info(f"   Average Cost per Query: ${overview['average_cost_per_query']}")
        logger.info(f"   Average Time per Query: {overview['average_time_per_query']}s")
        
        # Graph Building Summary
        graph_summary = report['graph_building_summary']
        if 'status' not in graph_summary:
            logger.info("\n🏗️ GRAPH BUILDING SUMMARY:")
            time_breakdown = graph_summary['time_breakdown']
            cost_breakdown = graph_summary['cost_breakdown']
            logger.info(f"   Time - Chunking: {time_breakdown['chunking_time']}s")
            logger.info(f"   Time - Entity Extraction: {time_breakdown['entity_extraction_time']}s")
            logger.info(f"   Time - Relationship Extraction: {time_breakdown['relationship_extraction_time']}s")
            logger.info(f"   Time - Graph Construction: {time_breakdown['graph_construction_time']}s")
            logger.info(f"   Time - Total: {time_breakdown['total_time']}s")
            logger.info(f"   Tokens - Total: {cost_breakdown['total_tokens']}")
            logger.info(f"   Cost - Total: ${cost_breakdown['total_cost_usd']}")
        
        # Query Processing Summary
        runtime_summary = report['memos_runtime_summary']
        if 'status' not in runtime_summary:
            logger.info("\n📊 MEMOS-STYLE QUERY SUMMARY:")
            memos_breakdown = runtime_summary['memos_breakdown']
            logger.info(f"   Total Queries: {runtime_summary['total_queries']}")
            logger.info(f"   MemOS Context Tokens - Avg: {memos_breakdown['context_tokens_avg']}")
            logger.info(f"   MemOS Duration - Response Avg: {memos_breakdown['response_duration_ms_avg']} ms")
            logger.info(f"   MemOS Duration - Search Avg: {memos_breakdown['search_duration_ms_avg']} ms")
            logger.info(f"   MemOS Duration - Total Avg: {memos_breakdown['total_duration_ms_avg']} ms")
            logger.info(f"   Cost - Total: ${memos_breakdown['total_query_cost_usd']}")
        
        # Evaluation Summary
        eval_summary = report['evaluation_summary']
        if 'status' not in eval_summary:
            logger.info("\n📈 EVALUATION SUMMARY:")
            logger.info(f"   Accuracy: {eval_summary['accuracy']:.2%}")
            logger.info(f"   Correct Answers: {eval_summary['correct_answers']}/{eval_summary['total_answers']}")
            logger.info(f"   Answer Rate: {eval_summary['answer_rate']:.2%}")
        
        # Cost Analysis
        cost_analysis = report['cost_analysis']
        logger.info("\n💰 COST ANALYSIS:")
        logger.info(f"   Graph Building Cost: ${cost_analysis['graph_building_cost']}")
        logger.info(f"   Query Processing Cost: ${cost_analysis['query_processing_cost']}")
        logger.info(f"   Total System Cost: ${cost_analysis['total_system_cost']}")
        logger.info(f"   Cost per Query: ${cost_analysis['cost_efficiency']['cost_per_query']}")
        
        # Performance Analysis
        perf_analysis = report['performance_analysis']
        logger.info("\n⚡ PERFORMANCE ANALYSIS:")
        logger.info(f"   Graph Building Time: {perf_analysis['graph_building_time']}s")
        logger.info(f"   Query Processing Time: {perf_analysis['query_processing_time']}s")
        logger.info(f"   Total System Time: {perf_analysis['total_system_time']}s")
        logger.info(f"   Queries per Minute: {perf_analysis['throughput_metrics']['queries_per_minute']}")
        
        # Recommendations
        recommendations = report['recommendations']
        if recommendations['total_recommendations'] > 0:
            logger.info("\n💡 OPTIMIZATION RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations['recommendations'], 1):
                logger.info(f"   {i}. [{rec['type'].upper()}] {rec['message']} (Impact: {rec['impact']})")
        
        logger.info("=" * 100)
        
        return report
