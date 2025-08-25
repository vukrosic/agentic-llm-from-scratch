#!/usr/bin/env python3
"""
Adaptable GPU utilization monitor for distributed training and general use
"""

import argparse
import json
import csv
import logging
import os
import subprocess
import sys
import threading
import time
import signal
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum


class OutputFormat(Enum):
    """Available output formats"""
    CONSOLE = "console"
    JSON = "json"
    CSV = "csv"
    LOG = "log"


class GPUVendor(Enum):
    """Supported GPU vendors"""
    NVIDIA = "nvidia"
    AMD = "amd"
    INTEL = "intel"


@dataclass
class GPUStats:
    """GPU statistics data structure"""
    index: int
    name: str
    utilization: float
    memory_used: int
    memory_total: int
    temperature: float
    power_draw: Optional[float] = None
    power_limit: Optional[float] = None
    fan_speed: Optional[float] = None
    timestamp: Optional[str] = None
    
    def memory_percent(self) -> float:
        """Calculate memory usage percentage"""
        return (self.memory_used / self.memory_total) * 100 if self.memory_total > 0 else 0.0


@dataclass
class MonitorConfig:
    """Configuration for GPU monitoring"""
    interval: float = 2.0
    output_format: OutputFormat = OutputFormat.CONSOLE
    output_file: Optional[str] = None
    log_level: str = "INFO"
    vendor: GPUVendor = GPUVendor.NVIDIA
    specific_gpus: Optional[List[int]] = None
    alert_thresholds: Optional[Dict[str, float]] = None
    show_detailed_stats: bool = False
    continuous_mode: bool = True
    max_samples: Optional[int] = None
    enable_alerts: bool = True
    alert_callbacks: Optional[List[Callable]] = None


class GPUStatsProvider(ABC):
    """Abstract base class for GPU statistics providers"""
    
    @abstractmethod
    def get_gpu_stats(self, specific_gpus: Optional[List[int]] = None) -> List[GPUStats]:
        """Get GPU statistics"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider is available on the system"""
        pass


class NvidiaStatsProvider(GPUStatsProvider):
    """NVIDIA GPU statistics provider using nvidia-smi"""
    
    def __init__(self, detailed: bool = False):
        self.detailed = detailed
        self._base_query = "index,name,utilization.gpu,memory.used,memory.total,temperature.gpu"
        if detailed:
            self._base_query += ",power.draw,power.limit,fan.speed"
    
    def get_gpu_stats(self, specific_gpus: Optional[List[int]] = None) -> List[GPUStats]:
        """Get NVIDIA GPU statistics using nvidia-smi"""
        try:
            cmd = [
                'nvidia-smi',
                f'--query-gpu={self._base_query}',
                '--format=csv,noheader,nounits'
            ]
            
            if specific_gpus:
                gpu_ids = ','.join(map(str, specific_gpus))
                cmd.extend(['--id', gpu_ids])
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                return self._parse_nvidia_output(result.stdout)
            else:
                logging.error(f"nvidia-smi failed with return code {result.returncode}: {result.stderr}")
        except subprocess.TimeoutExpired:
            logging.error("nvidia-smi command timed out")
        except Exception as e:
            logging.error(f"Error getting NVIDIA GPU stats: {e}")
        
        return []
    
    def _parse_nvidia_output(self, output: str) -> List[GPUStats]:
        """Parse nvidia-smi output into GPUStats objects"""
        stats = []
        lines = output.strip().split('\n')
        
        for line in lines:
            if not line.strip():
                continue
                
            parts = [p.strip() for p in line.split(',')]
            expected_parts = 6 + (3 if self.detailed else 0)
            
            if len(parts) >= expected_parts:
                try:
                    gpu_stat = GPUStats(
                        index=int(parts[0]),
                        name=parts[1],
                        utilization=float(parts[2]) if parts[2] != 'N/A' else 0.0,
                        memory_used=int(parts[3]) if parts[3] != 'N/A' else 0,
                        memory_total=int(parts[4]) if parts[4] != 'N/A' else 0,
                        temperature=float(parts[5]) if parts[5] != 'N/A' else 0.0,
                        timestamp=datetime.now().isoformat()
                    )
                    
                    if self.detailed and len(parts) >= 9:
                        gpu_stat.power_draw = float(parts[6]) if parts[6] != 'N/A' else None
                        gpu_stat.power_limit = float(parts[7]) if parts[7] != 'N/A' else None
                        gpu_stat.fan_speed = float(parts[8]) if parts[8] != 'N/A' else None
                    
                    stats.append(gpu_stat)
                except (ValueError, IndexError) as e:
                    logging.warning(f"Failed to parse GPU stats line '{line}': {e}")
                    continue
        
        return stats
    
    def is_available(self) -> bool:
        """Check if nvidia-smi is available"""
        try:
            subprocess.run(['nvidia-smi', '--version'], 
                          capture_output=True, check=True, timeout=5)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return False


class AMDStatsProvider(GPUStatsProvider):
    """AMD GPU statistics provider using rocm-smi"""
    
    def get_gpu_stats(self, specific_gpus: Optional[List[int]] = None) -> List[GPUStats]:
        """Get AMD GPU statistics using rocm-smi"""
        # TODO: Implement AMD GPU monitoring
        logging.warning("AMD GPU monitoring not yet implemented")
        return []
    
    def is_available(self) -> bool:
        """Check if rocm-smi is available"""
        try:
            subprocess.run(['rocm-smi', '--version'], 
                          capture_output=True, check=True, timeout=5)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return False


class OutputHandler(ABC):
    """Abstract base class for output handlers"""
    
    @abstractmethod
    def write_stats(self, stats: List[GPUStats], summary: Optional[Dict] = None) -> None:
        """Write GPU statistics"""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close output handler"""
        pass


class ConsoleOutputHandler(OutputHandler):
    """Console output handler with colored output"""
    
    def __init__(self, show_detailed: bool = False):
        self.show_detailed = show_detailed
    
    def write_stats(self, stats: List[GPUStats], summary: Optional[Dict] = None) -> None:
        """Write stats to console with color coding"""
        if not stats:
            print("No GPU data available")
            return
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"\n[{timestamp}] GPU Status:")
        
        for gpu in stats:
            util = gpu.utilization
            mem_percent = gpu.memory_percent()
            
            # Color coding for utilization
            util_color = self._get_utilization_color(util)
            
            # Basic stats
            print(f"  GPU {gpu.index}: {util_color} {util:5.1f}% util | "
                  f"{gpu.memory_used:5d}/{gpu.memory_total:5d}MB ({mem_percent:4.1f}%) | "
                  f"{gpu.temperature:4.1f}°C | {gpu.name}")
            
            # Detailed stats if enabled
            if self.show_detailed:
                if gpu.power_draw is not None:
                    power_info = f"{gpu.power_draw:.1f}W"
                    if gpu.power_limit:
                        power_info += f"/{gpu.power_limit:.1f}W"
                    print(f"    Power: {power_info}")
                
                if gpu.fan_speed is not None:
                    print(f"    Fan: {gpu.fan_speed:.0f}%")
        
        # Summary stats for multiple GPUs
        if summary and len(stats) > 1:
            print(f"  📊 {summary}")
    
    def _get_utilization_color(self, util: float) -> str:
        """Get color emoji for utilization level"""
        if util < 20:
            return "🔴"  # Red for low utilization
        elif util < 60:
            return "🟡"  # Yellow for medium
        else:
            return "🟢"  # Green for high
    
    def close(self) -> None:
        """Nothing to close for console output"""
        pass


class JSONOutputHandler(OutputHandler):
    """JSON output handler"""
    
    def __init__(self, output_file: Optional[str] = None):
        self.output_file = output_file
        self.file_handle = None
        if output_file:
            self.file_handle = open(output_file, 'w')
    
    def write_stats(self, stats: List[GPUStats], summary: Optional[Dict] = None) -> None:
        """Write stats as JSON"""
        data = {
            "timestamp": datetime.now().isoformat(),
            "gpus": [asdict(gpu) for gpu in stats],
            "summary": summary
        }
        
        json_str = json.dumps(data, indent=2)
        
        if self.file_handle:
            self.file_handle.write(json_str + '\n')
            self.file_handle.flush()
        else:
            print(json_str)
    
    def close(self) -> None:
        """Close file handle"""
        if self.file_handle:
            self.file_handle.close()


class CSVOutputHandler(OutputHandler):
    """CSV output handler"""
    
    def __init__(self, output_file: Optional[str] = None):
        self.output_file = output_file
        self.file_handle = None
        self.csv_writer = None
        self.header_written = False
        
        if output_file:
            self.file_handle = open(output_file, 'w', newline='')
            self.csv_writer = csv.writer(self.file_handle)
    
    def write_stats(self, stats: List[GPUStats], summary: Optional[Dict] = None) -> None:
        """Write stats as CSV"""
        if not stats:
            return
        
        # Write header if first time
        if not self.header_written:
            header = list(asdict(stats[0]).keys())
            if self.csv_writer:
                self.csv_writer.writerow(header)
            else:
                print(','.join(header))
            self.header_written = True
        
        # Write data rows
        for gpu in stats:
            row = list(asdict(gpu).values())
            if self.csv_writer:
                self.csv_writer.writerow(row)
                self.file_handle.flush()
            else:
                print(','.join(map(str, row)))
    
    def close(self) -> None:
        """Close file handle"""
        if self.file_handle:
            self.file_handle.close()


class AdaptableGPUMonitor:
    """Adaptable GPU monitor with configurable options"""
    
    def __init__(self, config: MonitorConfig):
        self.config = config
        self.running = False
        self.thread = None
        self.sample_count = 0
        self.stats_history = []
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, config.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize GPU stats provider
        self.stats_provider = self._get_stats_provider()
        
        # Initialize output handler
        self.output_handler = self._get_output_handler()
        
        # Alert thresholds
        self.alert_thresholds = config.alert_thresholds or {
            'utilization_imbalance': 30.0,
            'temperature_warning': 80.0,
            'temperature_critical': 90.0,
            'memory_warning': 90.0
        }
    
    def _get_stats_provider(self) -> GPUStatsProvider:
        """Get appropriate stats provider based on configuration"""
        if self.config.vendor == GPUVendor.NVIDIA:
            provider = NvidiaStatsProvider(detailed=self.config.show_detailed_stats)
        elif self.config.vendor == GPUVendor.AMD:
            provider = AMDStatsProvider()
        else:
            raise ValueError(f"Unsupported GPU vendor: {self.config.vendor}")
        
        if not provider.is_available():
            raise RuntimeError(f"{self.config.vendor.value} GPU tools not available")
        
        return provider
    
    def _get_output_handler(self) -> OutputHandler:
        """Get appropriate output handler based on configuration"""
        if self.config.output_format == OutputFormat.CONSOLE:
            return ConsoleOutputHandler(show_detailed=self.config.show_detailed_stats)
        elif self.config.output_format == OutputFormat.JSON:
            return JSONOutputHandler(self.config.output_file)
        elif self.config.output_format == OutputFormat.CSV:
            return CSVOutputHandler(self.config.output_file)
        elif self.config.output_format == OutputFormat.LOG:
            # For log format, we'll use JSON but write to logger
            return JSONOutputHandler(self.config.output_file)
        else:
            raise ValueError(f"Unsupported output format: {self.config.output_format}")
    
    def get_gpu_stats(self) -> List[GPUStats]:
        """Get current GPU statistics"""
        return self.stats_provider.get_gpu_stats(self.config.specific_gpus)
    
    def calculate_summary(self, stats: List[GPUStats]) -> Dict:
        """Calculate summary statistics"""
        if not stats:
            return {}
        
        utilizations = [gpu.utilization for gpu in stats]
        temperatures = [gpu.temperature for gpu in stats]
        memory_percents = [gpu.memory_percent() for gpu in stats]
        
        summary = {
            'total_gpus': len(stats),
            'avg_utilization': sum(utilizations) / len(utilizations),
            'max_utilization': max(utilizations),
            'min_utilization': min(utilizations),
            'utilization_imbalance': max(utilizations) - min(utilizations),
            'avg_temperature': sum(temperatures) / len(temperatures),
            'max_temperature': max(temperatures),
            'avg_memory_usage': sum(memory_percents) / len(memory_percents),
            'max_memory_usage': max(memory_percents)
        }
        
        return summary
    
    def check_alerts(self, stats: List[GPUStats], summary: Dict) -> List[str]:
        """Check for alert conditions"""
        alerts = []
        
        if not self.config.enable_alerts:
            return alerts
        
        # Check utilization imbalance
        if len(stats) > 1:
            imbalance = summary.get('utilization_imbalance', 0)
            if imbalance > self.alert_thresholds['utilization_imbalance']:
                alerts.append(f"HIGH UTILIZATION IMBALANCE: {imbalance:.1f}% difference")
        
        # Check temperatures
        for gpu in stats:
            if gpu.temperature > self.alert_thresholds['temperature_critical']:
                alerts.append(f"CRITICAL TEMPERATURE: GPU {gpu.index} at {gpu.temperature:.1f}°C")
            elif gpu.temperature > self.alert_thresholds['temperature_warning']:
                alerts.append(f"High temperature warning: GPU {gpu.index} at {gpu.temperature:.1f}°C")
        
        # Check memory usage
        for gpu in stats:
            mem_percent = gpu.memory_percent()
            if mem_percent > self.alert_thresholds['memory_warning']:
                alerts.append(f"High memory usage: GPU {gpu.index} at {mem_percent:.1f}%")
        
        return alerts
    
    def monitor_loop(self) -> None:
        """Main monitoring loop"""
        self.logger.info("GPU monitoring started")
        
        while self.running:
            try:
                # Get GPU stats
                stats = self.get_gpu_stats()
                
                if stats:
                    # Calculate summary
                    summary = self.calculate_summary(stats)
                    
                    # Check for alerts
                    alerts = self.check_alerts(stats, summary)
                    
                    # Store history if needed
                    if hasattr(self, 'store_history') and self.store_history:
                        self.stats_history.append({
                            'timestamp': datetime.now().isoformat(),
                            'stats': stats,
                            'summary': summary
                        })
                    
                    # Output stats
                    self.output_handler.write_stats(stats, summary)
                    
                    # Handle alerts
                    for alert in alerts:
                        self.logger.warning(f"ALERT: {alert}")
                        if self.config.output_format == OutputFormat.CONSOLE:
                            print(f"  ⚠️  {alert}")
                        
                        # Call alert callbacks if configured
                        if self.config.alert_callbacks:
                            for callback in self.config.alert_callbacks:
                                try:
                                    callback(alert, stats, summary)
                                except Exception as e:
                                    self.logger.error(f"Alert callback failed: {e}")
                    
                    self.sample_count += 1
                    
                    # Check if we've reached max samples
                    if (self.config.max_samples and 
                        self.sample_count >= self.config.max_samples):
                        self.logger.info(f"Reached max samples ({self.config.max_samples})")
                        break
                
                else:
                    self.logger.warning("No GPU data available")
                
                # Sleep for the specified interval
                time.sleep(self.config.interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                if not self.config.continuous_mode:
                    break
                time.sleep(self.config.interval)
        
        self.logger.info("GPU monitoring stopped")
    
    def start(self) -> None:
        """Start monitoring in background thread"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self.monitor_loop, daemon=True)
            self.thread.start()
            self.logger.info("GPU monitor started in background")
    
    def stop(self) -> None:
        """Stop monitoring"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        self.output_handler.close()
        self.logger.info("GPU monitor stopped")
    
    def run_once(self) -> List[GPUStats]:
        """Run monitoring once and return stats"""
        stats = self.get_gpu_stats()
        if stats:
            summary = self.calculate_summary(stats)
            self.output_handler.write_stats(stats, summary)
        return stats
    
    def export_history(self, filename: str) -> None:
        """Export monitoring history to file"""
        if not hasattr(self, 'stats_history') or not self.stats_history:
            self.logger.warning("No history to export")
            return
        
        with open(filename, 'w') as f:
            json.dump(self.stats_history, f, indent=2, default=str)
        
        self.logger.info(f"History exported to {filename}")


def create_config_from_args(args) -> MonitorConfig:
    """Create MonitorConfig from command line arguments"""
    config = MonitorConfig(
        interval=args.interval,
        output_format=OutputFormat(args.output_format),
        output_file=args.output_file,
        log_level=args.log_level,
        vendor=GPUVendor(args.vendor),
        specific_gpus=args.gpus,
        show_detailed_stats=args.detailed,
        continuous_mode=not args.once,
        max_samples=args.max_samples,
        enable_alerts=not args.no_alerts
    )
    
    # Custom alert thresholds
    if args.alert_thresholds:
        try:
            config.alert_thresholds = json.loads(args.alert_thresholds)
        except json.JSONDecodeError:
            logging.warning("Invalid alert thresholds JSON, using defaults")
    
    return config


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    print("\n\n🛑 Monitoring stopped by user")
    sys.exit(0)


def main():
    """Main entry point with command line argument parsing"""
    parser = argparse.ArgumentParser(
        description="Adaptable GPU utilization monitor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Basic monitoring
  %(prog)s --interval 1 --detailed           # Detailed stats every second
  %(prog)s --output-format json --output-file gpu_stats.json
  %(prog)s --gpus 0,1 --once                 # Monitor specific GPUs once
  %(prog)s --vendor amd                      # Monitor AMD GPUs
  %(prog)s --alert-thresholds '{"temperature_warning": 75}'
        """
    )
    
    parser.add_argument('--interval', type=float, default=2.0,
                       help='Monitoring interval in seconds (default: 2.0)')
    
    parser.add_argument('--output-format', choices=['console', 'json', 'csv', 'log'],
                       default='console', help='Output format (default: console)')
    
    parser.add_argument('--output-file', type=str,
                       help='Output file path (for json/csv formats)')
    
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level (default: INFO)')
    
    parser.add_argument('--vendor', choices=['nvidia', 'amd', 'intel'],
                       default='nvidia', help='GPU vendor (default: nvidia)')
    
    parser.add_argument('--gpus', type=str,
                       help='Comma-separated list of GPU indices to monitor (e.g., "0,1,2")')
    
    parser.add_argument('--detailed', action='store_true',
                       help='Show detailed GPU statistics (power, fan speed, etc.)')
    
    parser.add_argument('--once', action='store_true',
                       help='Run once and exit (instead of continuous monitoring)')
    
    parser.add_argument('--max-samples', type=int,
                       help='Maximum number of samples to collect before stopping')
    
    parser.add_argument('--no-alerts', action='store_true',
                       help='Disable alert checking')
    
    parser.add_argument('--alert-thresholds', type=str,
                       help='Custom alert thresholds as JSON string')
    
    parser.add_argument('--version', action='version', version='%(prog)s 2.0')
    
    args = parser.parse_args()
    
    # Parse GPU list
    if args.gpus:
        try:
            args.gpus = [int(x.strip()) for x in args.gpus.split(',')]
        except ValueError:
            parser.error("Invalid GPU list format. Use comma-separated integers (e.g., '0,1,2')")
    
    # Create configuration
    config = create_config_from_args(args)
    
    try:
        # Create and start monitor
        monitor = AdaptableGPUMonitor(config)
        
        if args.once:
            # Run once and exit
            stats = monitor.run_once()
            print(f"\nMonitored {len(stats)} GPU(s)")
        else:
            # Set up signal handler for graceful exit
            signal.signal(signal.SIGINT, signal_handler)
            
            print("🚀 Adaptable GPU Monitor")
            print("Press Ctrl+C to stop monitoring")
            
            monitor.start()
            
            try:
                # Keep main thread alive
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                monitor.stop()
                print("\n🛑 Monitoring stopped")
    
    except Exception as e:
        logging.error(f"Failed to start GPU monitor: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()