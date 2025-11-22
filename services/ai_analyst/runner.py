"""
AI Analyst Service Runner
Schedules per-signal context generation and daily summaries
VERSION: 2.1 - Module reload trigger after clearing Python cache
"""

import os
import sys
import time
import logging
import yaml
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional

if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from services.ai_analyst.ai_client import AIClient
from services.ai_analyst.render import ResponseRenderer
from services.ai_analyst.sinks import OutputSink
from services.ai_analyst.health import HealthMonitor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    encoding='utf-8',
    force=True
)
logger = logging.getLogger(__name__)

logging.getLogger('openai').setLevel(logging.INFO)
logging.getLogger('httpx').setLevel(logging.INFO)
logging.getLogger('services.ai_analyst').setLevel(logging.INFO)


class AIAnalystService:
    """Main AI Analyst service orchestrator"""
    
    def __init__(self, config_path: str = 'config.yaml'):
        logger.info("Initializing AI Analyst Service...")
        
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.ai_config = self.config.get('ai_analyst', {})
        self.feature_flags = self.config.get('feature_flags', {})
        
        if not self.feature_flags.get('enable_ai_analyst', False):
            logger.warning("AI Analyst is disabled in config")
            self.enabled = False
            return
        
        self.enabled = True
        self.per_signal_enabled = self.ai_config.get('per_signal_enabled', True)
        self.daily_summary_enabled = self.ai_config.get('daily_summary_enabled', True)
        
        try:
            self.ai_client = AIClient(self.ai_config)
            self.renderer = ResponseRenderer()
            self.sink = OutputSink()
            self.health = HealthMonitor()
            
            self._load_prompts()
            
            logger.info("AI Analyst Service initialized successfully")
        
        except Exception as e:
            logger.error(f"Failed to initialize AI Analyst: {e}")
            self.enabled = False
    
    def _load_prompts(self):
        """Load system prompts from files with ASCII sanitization"""
        with open('services/ai_analyst/prompts/market_context.md') as f:
            raw_prompt = f.read()
            self.market_context_prompt = raw_prompt.encode('ascii', errors='replace').decode('ascii')
        
        with open('services/ai_analyst/prompts/daily_summary.md') as f:
            raw_prompt = f.read()
            self.daily_summary_prompt = raw_prompt.encode('ascii', errors='replace').decode('ascii')
        
        with open('services/ai_analyst/prompts/effectiveness_query.md') as f:
            raw_prompt = f.read()
            self.effectiveness_query_prompt = raw_prompt.encode('ascii', errors='replace').decode('ascii')
        
        with open('services/ai_analyst/prompts/signal_query.md') as f:
            raw_prompt = f.read()
            self.signal_query_prompt = raw_prompt.encode('ascii', errors='replace').decode('ascii')
    
    def generate_signal_context(
        self,
        symbol: str,
        verdict: str,
        confidence: float,
        features: Dict[str, Any],
        signal_id: Optional[str] = None,
        message_id: Optional[int] = None
    ) -> bool:
        """
        Generate and send AI context for a signal
        
        Args:
            symbol: Trading symbol
            verdict: BUY/SELL
            confidence: Signal confidence
            features: Dict of feature values
            signal_id: Unique signal ID
            message_id: Telegram message ID to reply to
        
        Returns:
            True if successful
        """
        if not self.enabled or not self.per_signal_enabled:
            return False
        
        try:
            user_prompt = self._build_signal_prompt(symbol, verdict, confidence, features)
            
            result = self.ai_client.get_completion(
                system_prompt=self.market_context_prompt,
                user_prompt=user_prompt,
                max_tokens=200,
                use_cache=True
            )
            
            self.health.record_call(
                success=(result['error'] is None),
                error=result['error']
            )
            
            if result['error']:
                logger.error(f"AI call failed: {result['error']}")
                self.sink.log_signal_analysis(
                    symbol=symbol,
                    signal_id=signal_id or f"{symbol}_{int(time.time())}",
                    verdict=verdict,
                    bot_confidence=confidence,
                    ai_confidence=None,
                    ai_ttl_minutes=None,
                    ai_target_pct=None,
                    features_used=','.join(features.keys())[:100],
                    ai_result=result,
                    summary="",
                    reasoning=""
                )
                return False
            
            parsed = self.ai_client.parse_ai_response(result['response'])
            
            if parsed['parse_error']:
                logger.warning(f"JSON parse failed: {parsed['parse_error']}, using raw text fallback")
            
            bot_ttl = features.get('ttl_minutes')
            bot_target = features.get('target_pct')
            
            formatted_text = self.renderer.render_market_context(
                bot_confidence=confidence,
                ai_confidence=parsed['ai_confidence'],
                ai_ttl_minutes=parsed.get('ai_ttl_minutes'),
                ai_target_pct=parsed.get('ai_target_pct'),
                bot_ttl_minutes=bot_ttl,
                bot_target_pct=bot_target,
                reasoning=parsed['reasoning']
            )
            
            summary = self.renderer.truncate_summary(parsed['reasoning'] or result['response'], 100)
            
            self.sink.log_signal_analysis(
                symbol=symbol,
                signal_id=signal_id or f"{symbol}_{int(time.time())}",
                verdict=verdict,
                bot_confidence=confidence,
                ai_confidence=parsed['ai_confidence'],
                ai_ttl_minutes=parsed.get('ai_ttl_minutes'),
                ai_target_pct=parsed.get('ai_target_pct'),
                features_used=','.join(features.keys())[:100],
                ai_result=result,
                summary=summary,
                reasoning=parsed['reasoning'] or ""
            )
            
            if formatted_text:
                self.sink.send_signal_context(formatted_text, reply_to_message_id=message_id)
            
            logger.info(f"Generated context for {symbol} {verdict}")
            return True
        
        except Exception as e:
            logger.error(f"Error generating signal context: {e}")
            self.health.record_call(success=False, error=str(e))
            return False
    
    def _build_signal_prompt(
        self,
        symbol: str,
        verdict: str,
        confidence: float,
        features: Dict[str, Any]
    ) -> str:
        """Build user prompt from signal features"""
        
        symbol = str(symbol).encode('ascii', errors='replace').decode('ascii')
        verdict = str(verdict).encode('ascii', errors='replace').decode('ascii')
        
        prompt_parts = [
            f"Symbol: {symbol}",
            f"Signal: {verdict}",
            f"Confidence: {confidence:.2f}",
            ""
        ]
        
        feature_order = [
            'cvd', 'oi_change_pct', 'dev_sigma', 'volume_mult', 'liq_ratio',
            'funding_rate', 'basis_pct', 'rsi', 'adx14', 'regime',
            'market_strength', 'zcvd', 'doi_pct', 'dev_sigma_uif', 'rsi_dist'
        ]
        
        prompt_parts.append("Key Features:")
        for feat in feature_order:
            if feat in features and features[feat] is not None:
                val = features[feat]
                feat_safe = str(feat).encode('ascii', errors='replace').decode('ascii')
                if isinstance(val, float):
                    prompt_parts.append(f"- {feat_safe}: {val:.2f}")
                else:
                    val_safe = str(val).encode('ascii', errors='replace').decode('ascii')
                    prompt_parts.append(f"- {feat_safe}: {val_safe}")
        
        prompt = '\n'.join(prompt_parts)
        return prompt.encode('ascii', errors='replace').decode('ascii')
    
    def generate_daily_summary(self, date_str: Optional[str] = None) -> bool:
        """
        Generate and send daily AI summary
        
        Args:
            date_str: Date string (YYYY-MM-DD), defaults to yesterday
        
        Returns:
            True if successful
        """
        if not self.enabled or not self.daily_summary_enabled:
            return False
        
        if date_str is None:
            yesterday = datetime.now(timezone.utc) - timedelta(days=1)
            date_str = yesterday.strftime('%Y-%m-%d')
        
        try:
            logger.info(f"Generating daily summary for {date_str}")
            
            summary_data = self._load_daily_data(date_str)
            
            if not summary_data:
                logger.warning(f"No data found for {date_str}")
                return False
            
            user_prompt = self._build_summary_prompt(date_str, summary_data)
            
            result = self.ai_client.get_completion(
                system_prompt=self.daily_summary_prompt,
                user_prompt=user_prompt,
                max_tokens=800,
                use_cache=False
            )
            
            self.health.record_call(
                success=(result['error'] is None),
                error=result['error']
            )
            
            if result['error']:
                logger.error(f"Daily summary AI call failed: {result['error']}")
                return False
            
            formatted_html = self.renderer.render_daily_summary(result['response'], date_str)
            
            self.sink.save_daily_summary_file(result['response'], date_str.replace('-', ''))
            
            self.sink.send_daily_summary(formatted_html)
            
            logger.info(f"Daily summary generated and sent for {date_str}")
            return True
        
        except Exception as e:
            logger.error(f"Error generating daily summary: {e}")
            self.health.record_call(success=False, error=str(e))
            return False
    
    def _load_daily_data(self, date_str: str) -> Optional[Dict[str, Any]]:
        """Load and aggregate data for a specific date"""
        try:
            import csv
            
            effectiveness_file = 'effectiveness_log.csv'
            if not os.path.exists(effectiveness_file):
                return None
            
            signals = []
            with open(effectiveness_file) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get('timestamp_sent', '').startswith(date_str):
                        signals.append(row)
            
            if not signals:
                return None
            
            wins = sum(1 for s in signals if s.get('result') == 'WIN')
            losses = sum(1 for s in signals if s.get('result') == 'LOSS')
            total = wins + losses
            
            if total == 0:
                return None
            
            wr = wins / total * 100
            
            pnls = [float(s.get('profit_pct', 0)) for s in signals if s.get('result') in ['WIN', 'LOSS']]
            total_pnl = sum(pnls)
            avg_pnl = total_pnl / len(pnls) if pnls else 0
            
            return {
                'date': date_str,
                'total_signals': len(signals),
                'wins': wins,
                'losses': losses,
                'wr': wr,
                'total_pnl': total_pnl,
                'avg_pnl': avg_pnl,
                'signals': signals
            }
        
        except Exception as e:
            logger.error(f"Error loading daily data: {e}")
            return None
    
    def _build_summary_prompt(self, date_str: str, data: Dict[str, Any]) -> str:
        """Build prompt for daily summary"""
        
        prompt = f"""Date: {date_str}

Performance Metrics:
- Total Signals: {data['total_signals']}
- Win Rate: {data['wr']:.1f}% ({data['wins']}W / {data['losses']}L)
- Total PnL: {data['total_pnl']:+.2f}%
- Average PnL: {data['avg_pnl']:+.3f}%

Analyze what worked and what didn't. Identify feature patterns and provide actionable recommendations."""
        
        return prompt.encode('ascii', errors='replace').decode('ascii')
    
    def query_ai(self, question: str, user_id: Optional[str] = 'webhook') -> Dict[str, Any]:
        """
        Query AI for effectiveness analysis or signal insights
        
        Args:
            question: User's question
            user_id: Telegram user ID (for logging), defaults to 'webhook'
        
        Returns:
            {
                'success': bool,
                'answer': str,
                'query_type': str ('effectiveness' or 'signal'),
                'tokens_total': int,
                'cost_usd': float,
                'error': str or None
            }
        """
        if not self.enabled:
            return {
                'success': False,
                'answer': '',
                'query_type': 'unknown',
                'tokens_total': 0,
                'cost_usd': 0.0,
                'error': 'AI Analyst is disabled'
            }
        
        try:
            query_type = self._classify_query(question)
            logger.info(f"Query type: {query_type}")
            
            context_data = self._load_query_context(query_type, question)
            
            if query_type == 'effectiveness':
                system_prompt = self.effectiveness_query_prompt
            else:
                system_prompt = self.signal_query_prompt
            
            user_prompt = self._build_query_prompt(question, context_data, query_type)
            
            result = self.ai_client.get_completion(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=500,
                use_cache=False
            )
            
            if result['error']:
                self._log_query(
                    user_id=user_id,
                    question=question,
                    query_type=query_type,
                    tokens_in=0,
                    tokens_out=0,
                    cost_usd=0.0,
                    success=False,
                    error=result['error']
                )
                return {
                    'success': False,
                    'answer': '',
                    'query_type': query_type,
                    'tokens_total': 0,
                    'cost_usd': 0.0,
                    'error': result['error']
                }
            
            answer = result['response']
            tokens_total = result['tokens_in'] + result['tokens_out']
            
            self._log_query(
                user_id=user_id,
                question=question,
                query_type=query_type,
                tokens_in=result['tokens_in'],
                tokens_out=result['tokens_out'],
                cost_usd=result['cost_usd'],
                success=True,
                error=None
            )
            
            logger.info(f"Query answered successfully: {tokens_total} tokens, ${result['cost_usd']:.4f}")
            
            return {
                'success': True,
                'answer': answer,
                'query_type': query_type,
                'tokens_total': tokens_total,
                'cost_usd': result['cost_usd'],
                'error': None
            }
        
        except Exception as e:
            logger.error(f"Error in query_ai: {e}", exc_info=True)
            self._log_query(
                user_id=user_id,
                question=question,
                query_type='error',
                tokens_in=0,
                tokens_out=0,
                cost_usd=0.0,
                success=False,
                error=str(e)
            )
            return {
                'success': False,
                'answer': '',
                'query_type': 'error',
                'tokens_total': 0,
                'cost_usd': 0.0,
                'error': str(e)
            }
    
    def process_interactive_query(self, question: str) -> Dict[str, Any]:
        """
        Wrapper for webhook server to process user queries
        
        Args:
            question: User's natural language question
        
        Returns:
            Same as query_ai() with tokens_used alias for backward compatibility
        """
        result = self.query_ai(question, user_id='webhook')
        
        if result['success']:
            result['tokens_used'] = result['tokens_total']
        
        return result
    
    def _classify_query(self, question: str) -> str:
        """Classify query type based on keywords"""
        question_lower = question.lower()
        
        effectiveness_keywords = [
            'win rate', 'winrate', 'effectiveness', 'performance', 'profit', 'loss',
            'indicator', 'formula', 'improve', 'better', 'worse', 'drop', 'increase',
            'correlation', 'vwap', 'rsi', 'ema', 'adx', 'cvd', 'oi', 'compare',
            'last week', 'last month', 'yesterday', 'statistics', 'stats'
        ]
        
        for keyword in effectiveness_keywords:
            if keyword in question_lower:
                return 'effectiveness'
        
        return 'signal'
    
    def _load_query_context(self, query_type: str, question: str) -> Dict[str, Any]:
        """Load relevant context data for query"""
        import csv
        from collections import defaultdict
        
        context = {}
        
        try:
            if os.path.exists('effectiveness_log.csv'):
                signals = []
                with open('effectiveness_log.csv', 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        signals.append(row)
                
                # ALL-TIME statistics
                all_wins = sum(1 for s in signals if s.get('result') == 'WIN')
                all_losses = sum(1 for s in signals if s.get('result') == 'LOSS')
                all_cancelled = sum(1 for s in signals if s.get('result') == 'CANCELLED')
                all_total = all_wins + all_losses
                
                if all_total > 0:
                    context['alltime_win_rate'] = round(all_wins / all_total * 100, 1)
                    context['alltime_wins'] = all_wins
                    context['alltime_losses'] = all_losses
                    context['alltime_cancelled'] = all_cancelled
                    context['alltime_total_signals'] = len(signals)
                
                # RECENT (last 100) statistics
                recent_signals = signals[-100:] if len(signals) > 100 else signals
                
                wins = sum(1 for s in recent_signals if s.get('result') == 'WIN')
                losses = sum(1 for s in recent_signals if s.get('result') == 'LOSS')
                cancelled = sum(1 for s in recent_signals if s.get('result') == 'CANCELLED')
                total = wins + losses
                
                if total > 0:
                    context['win_rate'] = round(wins / total * 100, 1)
                    context['total_signals'] = len(recent_signals)
                    context['wins'] = wins
                    context['losses'] = losses
                    context['cancelled'] = cancelled
                    
                    profits = [float(s.get('profit_pct', 0)) for s in recent_signals if s.get('result') in ['WIN', 'LOSS']]
                    if profits:
                        context['avg_profit'] = round(sum(profits) / len(profits), 2)
                        context['total_profit'] = round(sum(profits), 2)
                    
                    by_symbol = defaultdict(lambda: {'wins': 0, 'losses': 0, 'profits': []})  # type: ignore
                    for s in recent_signals:
                        if s.get('result') in ['WIN', 'LOSS']:
                            symbol = s.get('symbol', '')
                            if s.get('result') == 'WIN':
                                by_symbol[symbol]['wins'] += 1  # type: ignore
                            else:
                                by_symbol[symbol]['losses'] += 1  # type: ignore
                            by_symbol[symbol]['profits'].append(float(s.get('profit_pct', 0)))  # type: ignore
                    
                    symbol_stats = {}
                    for sym, data in by_symbol.items():
                        total_sym = data['wins'] + data['losses']  # type: ignore
                        if total_sym > 0:
                            symbol_stats[sym] = {
                                'wr': round(data['wins'] / total_sym * 100, 1),  # type: ignore
                                'total': total_sym,
                                'avg_profit': round(sum(data['profits']) / len(data['profits']), 2) if data['profits'] else 0  # type: ignore
                            }
                    
                    context['by_symbol'] = symbol_stats
            
            # LOAD HISTORICAL INDICATOR DATA from analysis_log.csv
            if os.path.exists('analysis_log.csv'):
                analysis_rows = []
                with open('analysis_log.csv', 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        analysis_rows.append(row)
                
                # Take last 150 rows for historical analysis
                recent_analysis = analysis_rows[-150:] if len(analysis_rows) > 150 else analysis_rows
                
                if recent_analysis:
                    # Calculate indicator statistics
                    def safe_float(val, default=0.0):
                        try:
                            return float(val) if val else default
                        except:
                            return default
                    
                    rsi_values = [safe_float(r.get('rsi')) for r in recent_analysis if r.get('rsi')]
                    oi_changes = [safe_float(r.get('oi_change_pct')) for r in recent_analysis if r.get('oi_change_pct')]
                    cvd_values = [safe_float(r.get('cvd')) for r in recent_analysis if r.get('cvd')]
                    dev_sigma_values = [safe_float(r.get('dev_sigma')) for r in recent_analysis if r.get('dev_sigma')]
                    adx_values = [safe_float(r.get('adx14')) for r in recent_analysis if r.get('adx14')]
                    funding_values = [safe_float(r.get('funding_rate')) for r in recent_analysis if r.get('funding_rate')]
                    
                    context['historical_indicators'] = {
                        'data_points': len(recent_analysis),
                        'rsi': {
                            'avg': round(sum(rsi_values) / len(rsi_values), 1) if rsi_values else 0,
                            'min': round(min(rsi_values), 1) if rsi_values else 0,
                            'max': round(max(rsi_values), 1) if rsi_values else 0
                        },
                        'oi_change_pct': {
                            'avg': round(sum(oi_changes) / len(oi_changes), 2) if oi_changes else 0,
                            'min': round(min(oi_changes), 2) if oi_changes else 0,
                            'max': round(max(oi_changes), 2) if oi_changes else 0
                        },
                        'dev_sigma': {
                            'avg': round(sum(dev_sigma_values) / len(dev_sigma_values), 2) if dev_sigma_values else 0,
                            'min': round(min(dev_sigma_values), 2) if dev_sigma_values else 0,
                            'max': round(max(dev_sigma_values), 2) if dev_sigma_values else 0
                        },
                        'adx14': {
                            'avg': round(sum(adx_values) / len(adx_values), 1) if adx_values else 0,
                            'min': round(min(adx_values), 1) if adx_values else 0,
                            'max': round(max(adx_values), 1) if adx_values else 0
                        }
                    }
                    
                    # Regime distribution
                    regimes = [r.get('regime', 'unknown') for r in recent_analysis if r.get('regime')]
                    regime_counts = defaultdict(int)
                    for regime in regimes:
                        regime_counts[regime] += 1
                    context['regime_distribution'] = dict(regime_counts)
                    
                    # Correlate indicators with outcomes (if we have matched data)
                    # Find signals with matching timestamps in effectiveness_log
                    if os.path.exists('effectiveness_log.csv'):
                        with open('effectiveness_log.csv', 'r') as f:
                            eff_reader = csv.DictReader(f)
                            eff_signals = {r['timestamp_sent']: r for r in eff_reader if r.get('timestamp_sent')}
                        
                        # Compute indicator correlations with WIN/LOSS
                        win_rsi = []
                        loss_rsi = []
                        win_oi = []
                        loss_oi = []
                        win_dev_sigma = []
                        loss_dev_sigma = []
                        
                        for row in recent_analysis:
                            ts = row.get('timestamp')
                            if ts in eff_signals:
                                result = eff_signals[ts].get('result')
                                if result == 'WIN':
                                    win_rsi.append(safe_float(row.get('rsi')))
                                    win_oi.append(safe_float(row.get('oi_change_pct')))
                                    win_dev_sigma.append(safe_float(row.get('dev_sigma')))
                                elif result == 'LOSS':
                                    loss_rsi.append(safe_float(row.get('rsi')))
                                    loss_oi.append(safe_float(row.get('oi_change_pct')))
                                    loss_dev_sigma.append(safe_float(row.get('dev_sigma')))
                        
                        if win_rsi and loss_rsi:
                            context['indicator_correlations'] = {
                                'rsi': {
                                    'win_avg': round(sum(win_rsi) / len(win_rsi), 1),
                                    'loss_avg': round(sum(loss_rsi) / len(loss_rsi), 1)
                                },
                                'oi_change_pct': {
                                    'win_avg': round(sum(win_oi) / len(win_oi), 2) if win_oi else 0,
                                    'loss_avg': round(sum(loss_oi) / len(loss_oi), 2) if loss_oi else 0
                                },
                                'dev_sigma': {
                                    'win_avg': round(sum(win_dev_sigma) / len(win_dev_sigma), 2) if win_dev_sigma else 0,
                                    'loss_avg': round(sum(loss_dev_sigma) / len(loss_dev_sigma), 2) if loss_dev_sigma else 0
                                }
                            }
            
            # LOAD BTC PRICE CORRELATION DATA (optional - safe fallback if file missing)
            try:
                if os.path.exists('analysis/results/btc_corr_summary.json'):
                    import json
                    with open('analysis/results/btc_corr_summary.json', 'r') as f:
                        corr_data = json.load(f)
                        
                        correlations = {}
                        for coin in corr_data.get('correlations', []):
                            symbol = coin['symbol']
                            correlations[symbol] = {
                                'correlation': round(coin['correlation'], 3),
                                'lag_min': coin['optimal_lag_minutes'],
                                'similarity_pct': round(coin['directional_similarity_pct'], 1)
                            }
                        
                        context['btc_correlations'] = correlations
                else:
                    # Correlation data not yet generated - AI will respond without it
                    context['btc_correlations'] = None
            except Exception as e:
                logger.warning(f"Failed to load BTC correlation data: {e}")
                context['btc_correlations'] = None
            
            if os.path.exists('data/ai_reports.csv'):
                ai_analyses = []
                with open('data/ai_reports.csv', 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        ai_analyses.append(row)
                
                context['ai_analysis_count'] = len(ai_analyses)
                if ai_analyses:
                    recent_ai = ai_analyses[-20:]
                    context['recent_ai_samples'] = [
                        {
                            'symbol': a.get('symbol', ''),
                            'bot_conf': a.get('bot_confidence', ''),
                            'ai_conf': a.get('ai_confidence', '')
                        }
                        for a in recent_ai[-5:]
                    ]
        
        except Exception as e:
            logger.error(f"Error loading query context: {e}")
            context['error'] = str(e)
        
        return context
    
    def _build_query_prompt(self, question: str, context: Dict[str, Any], query_type: str) -> str:
        """Build user prompt with question and context"""
        prompt_parts = [
            f"USER QUESTION: {question}",
            "",
            "AVAILABLE DATA:",
            ""
        ]
        
        # ALL-TIME statistics (if available)
        if context.get('alltime_win_rate') is not None:
            prompt_parts.append(f"All-Time Performance ({context.get('alltime_total_signals', 0)} total signals):")
            prompt_parts.append(f"- Win Rate: {context['alltime_win_rate']}% ({context.get('alltime_wins', 0)}W / {context.get('alltime_losses', 0)}L / {context.get('alltime_cancelled', 0)}C)")
            prompt_parts.append("")
        
        # RECENT statistics (last 100)
        if context.get('win_rate') is not None:
            prompt_parts.append(f"Recent Performance (last {context.get('total_signals', 0)} signals):")
            prompt_parts.append(f"- Win Rate: {context['win_rate']}% ({context.get('wins', 0)}W / {context.get('losses', 0)}L / {context.get('cancelled', 0)}C)")
            prompt_parts.append(f"- Average Profit: {context.get('avg_profit', 0):+.2f}%")
            prompt_parts.append(f"- Total Profit: {context.get('total_profit', 0):+.2f}%")
            prompt_parts.append("")
        
        if context.get('by_symbol'):
            prompt_parts.append("Performance by Symbol:")
            for sym, stats in sorted(context['by_symbol'].items(), key=lambda x: x[1]['wr'], reverse=True)[:5]:
                prompt_parts.append(f"- {sym}: {stats['wr']}% WR ({stats['total']} signals), avg {stats['avg_profit']:+.2f}%")
            prompt_parts.append("")
        
        # HISTORICAL INDICATOR DATA (NEW!)
        if context.get('historical_indicators'):
            hist = context['historical_indicators']
            prompt_parts.append(f"Historical Indicator Trends (last {hist.get('data_points', 0)} data points):")
            
            if 'rsi' in hist:
                prompt_parts.append(f"- RSI: avg {hist['rsi']['avg']}, range [{hist['rsi']['min']}-{hist['rsi']['max']}]")
            
            if 'oi_change_pct' in hist:
                prompt_parts.append(f"- OI Change %: avg {hist['oi_change_pct']['avg']}, range [{hist['oi_change_pct']['min']}-{hist['oi_change_pct']['max']}]")
            
            if 'dev_sigma' in hist:
                prompt_parts.append(f"- VWAP Sigma: avg {hist['dev_sigma']['avg']}, range [{hist['dev_sigma']['min']}-{hist['dev_sigma']['max']}]")
            
            if 'adx14' in hist:
                prompt_parts.append(f"- ADX14: avg {hist['adx14']['avg']}, range [{hist['adx14']['min']}-{hist['adx14']['max']}]")
            
            prompt_parts.append("")
        
        # REGIME DISTRIBUTION
        if context.get('regime_distribution'):
            regime_dist = context['regime_distribution']
            prompt_parts.append("Market Regime Distribution:")
            for regime, count in sorted(regime_dist.items(), key=lambda x: x[1], reverse=True):
                prompt_parts.append(f"- {regime}: {count} occurrences")
            prompt_parts.append("")
        
        # INDICATOR CORRELATIONS WITH WIN/LOSS
        if context.get('indicator_correlations'):
            corr = context['indicator_correlations']
            prompt_parts.append("Indicator Correlations (Win vs Loss signals):")
            
            if 'rsi' in corr:
                prompt_parts.append(f"- RSI: WIN avg {corr['rsi']['win_avg']} vs LOSS avg {corr['rsi']['loss_avg']}")
            
            if 'oi_change_pct' in corr:
                prompt_parts.append(f"- OI Change: WIN avg {corr['oi_change_pct']['win_avg']}% vs LOSS avg {corr['oi_change_pct']['loss_avg']}%")
            
            if 'dev_sigma' in corr:
                prompt_parts.append(f"- VWAP Sigma: WIN avg {corr['dev_sigma']['win_avg']} vs LOSS avg {corr['dev_sigma']['loss_avg']}")
            
            prompt_parts.append("")
        
        # BTC PRICE CORRELATIONS (if available)
        btc_corrs = context.get('btc_correlations')
        
        # Guard: skip if not a dict (None, list, string, etc)
        if not isinstance(btc_corrs, dict):
            if btc_corrs is not None:
                logger.warning(f"BTC correlations data has unexpected type: {type(btc_corrs)}")
        # Guard: skip if empty dict
        elif not btc_corrs:
            pass
        # Safe to iterate: confirmed dict with items
        else:
            try:
                prompt_parts.append("BTC Price Correlations (log returns-based):")
                sorted_corrs = sorted(btc_corrs.items(), key=lambda x: x[1]['correlation'], reverse=True)
                for symbol, corr_data in sorted_corrs[:5]:
                    prompt_parts.append(f"- {symbol}: corr {corr_data['correlation']:+.3f}, lag {corr_data['lag_min']}min, similarity {corr_data['similarity_pct']}%")
                prompt_parts.append("")
            except Exception as e:
                logger.warning(f"Failed to format BTC correlations for prompt: {e}")
        
        if context.get('error'):
            prompt_parts.append(f"[Data loading warning: {context['error']}]")
        
        prompt = '\n'.join(prompt_parts)
        return prompt.encode('ascii', errors='replace').decode('ascii')
    
    def _log_query(self, user_id: str, question: str, query_type: str, 
                   tokens_in: int, tokens_out: int, cost_usd: float,
                   success: bool, error: Optional[str]):
        """Log query to CSV"""
        import csv
        
        try:
            log_file = 'data/ai_query_log.csv'
            file_exists = os.path.exists(log_file)
            
            os.makedirs('data', exist_ok=True)
            
            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                
                if not file_exists:
                    writer.writerow([
                        'timestamp', 'user_id', 'question', 'query_type',
                        'tokens_in', 'tokens_out', 'cost_usd', 'success', 'error'
                    ])
                
                writer.writerow([
                    datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
                    user_id,
                    question[:200],
                    query_type,
                    tokens_in,
                    tokens_out,
                    cost_usd,
                    'yes' if success else 'no',
                    error or ''
                ])
        
        except Exception as e:
            logger.error(f"Failed to log query: {e}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get service health status"""
        if not self.enabled:
            return {'status': 'DISABLED'}
        
        health_status = self.health.get_status()
        ai_stats = self.ai_client.get_stats()
        
        return {
            **health_status,
            **ai_stats
        }
    
    def run_scheduler(self):
        """Main scheduler loop"""
        if not self.enabled:
            logger.warning("AI Analyst is disabled, exiting")
            return
        
        logger.info("Starting AI Analyst scheduler...")
        
        daily_summary_time = self.config.get('report', {}).get('time_utc', '18:59')
        
        last_summary_date = None
        
        while True:
            try:
                now = datetime.now(timezone.utc)
                current_time = now.strftime('%H:%M')
                current_date = now.strftime('%Y-%m-%d')
                
                if (current_time == daily_summary_time and 
                    current_date != last_summary_date and
                    self.daily_summary_enabled):
                    
                    logger.info(f"Triggering daily summary at {current_time}")
                    self.generate_daily_summary()
                    last_summary_date = current_date
                
                time.sleep(30)
            
            except KeyboardInterrupt:
                logger.info("Scheduler stopped by user")
                break
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                time.sleep(60)


def main():
    """Entry point"""
    logger.info("=" * 60)
    logger.info("AI Analyst Service Starting...")
    logger.info("=" * 60)
    
    service = AIAnalystService()
    
    if not service.enabled:
        logger.error("Service is disabled, check configuration and OPENAI_API_KEY")
        return
    
    status = service.get_health_status()
    logger.info(f"Health Status: {status}")
    
    service.run_scheduler()


if __name__ == '__main__':
    main()
