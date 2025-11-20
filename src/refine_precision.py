import re
import argparse
from collections import defaultdict
import json


def parse_log_file(log_file_path):
    """
    Parse the log file to extract self-refine decisions and final success/failure outcomes.
    
    Returns:
        dict: Contains statistics about self-refine performance
    """
    
    with open(log_file_path, 'r') as f:
        log_content = f.read()
    
    # Split log into subtasks based on "Scene ... Episode ... Subtask ..." pattern
    subtask_pattern = r'Scene \w+-\w+ Episode \d+ Subtask \d+/\d+'
    subtasks = re.split(subtask_pattern, log_content)[1:]  # Skip the first empty part
    
    stats = {
        'total_confirms': 0,
        'total_rejects': 0,
        'confirm_success': 0,
        'confirm_fail': 0,
        'reject_leading_to_success': 0,
        'reject_leading_to_fail': 0,
        'subtask_details': []
    }
    
    for i, subtask_content in enumerate(subtasks):
        subtask_stats = analyze_subtask(subtask_content, i + 1)
        
        # Aggregate statistics
        stats['total_confirms'] += subtask_stats['confirms']
        stats['total_rejects'] += subtask_stats['rejects']
        
        # Count outcomes for CONFIRM decisions
        if subtask_stats['final_decision'] == 'CONFIRM':
            if subtask_stats['final_success']:
                stats['confirm_success'] += 1
            else:
                stats['confirm_fail'] += 1
        
        # Track reject patterns (for additional analysis)
        if subtask_stats['rejects'] > 0:
            if subtask_stats['final_success']:
                stats['reject_leading_to_success'] += 1
            else:
                stats['reject_leading_to_fail'] += 1
        
        stats['subtask_details'].append(subtask_stats)
    
    return stats


def analyze_subtask(subtask_content, subtask_id):
    """
    Analyze a single subtask to extract self-refine decisions and final outcome.
    
    Args:
        subtask_content (str): Log content for one subtask
        subtask_id (int): Subtask identifier
    
    Returns:
        dict: Statistics for this subtask
    """
    
    # Find all self-refine decisions
    confirm_pattern = r'Self-refine: CONFIRMED choice for \w+ task'
    reject_pattern = r'Self-refine: REJECTED choice for \w+ task'
    
    confirms = len(re.findall(confirm_pattern, subtask_content))
    rejects = len(re.findall(reject_pattern, subtask_content))
    
    # Determine final decision type
    final_decision = None
    if confirms > 0:
        final_decision = 'CONFIRM'
    elif rejects > 0:
        # If only rejects, then either frontier was chosen or cycle was broken
        cycle_break_pattern = r'accepting as final choice to break cycle'
        if re.search(cycle_break_pattern, subtask_content):
            final_decision = 'CYCLE_BREAK'
        else:
            final_decision = 'FRONTIER_AFTER_REJECT'
    else:
        final_decision = 'NO_SELF_REFINE'
    
    # Determine final success/failure
    success_pattern = r'Success: agent reached the target viewpoint at distance'
    fail_pattern = r'Fail: agent failed to reach the target viewpoint at distance'
    
    final_success = bool(re.search(success_pattern, subtask_content))
    final_fail = bool(re.search(fail_pattern, subtask_content))
    
    # Extract task type
    task_type_pattern = r'Task type: (\w+)'
    task_type_match = re.search(task_type_pattern, subtask_content)
    task_type = task_type_match.group(1) if task_type_match else 'unknown'
    
    # Extract choice details if available
    snapshot_choice_pattern = r'Prediction: snapshot, (\d+)'
    frontier_choice_pattern = r'Prediction: frontier, (\d+)'
    
    choice_type = None
    if re.search(snapshot_choice_pattern, subtask_content):
        choice_type = 'snapshot'
    elif re.search(frontier_choice_pattern, subtask_content):
        choice_type = 'frontier'
    
    return {
        'subtask_id': subtask_id,
        'task_type': task_type,
        'confirms': confirms,
        'rejects': rejects,
        'final_decision': final_decision,
        'final_success': final_success,
        'final_fail': final_fail,
        'choice_type': choice_type
    }


def calculate_precision_metrics(stats):
    """
    Calculate precision and other relevant metrics for self-refine performance.
    
    Args:
        stats (dict): Statistics from parse_log_file
        
    Returns:
        dict: Calculated metrics
    """
    
    metrics = {}
    
    # Self-refine precision: when self-refine says CONFIRM, how often is it correct?
    total_final_confirms = stats['confirm_success'] + stats['confirm_fail']
    if total_final_confirms > 0:
        metrics['self_refine_precision'] = stats['confirm_success'] / total_final_confirms
    else:
        metrics['self_refine_precision'] = None
    
    # Overall accuracy: how often does the agent succeed?
    total_subtasks = len(stats['subtask_details'])
    total_successes = sum(1 for s in stats['subtask_details'] if s['final_success'])
    metrics['overall_success_rate'] = total_successes / total_subtasks if total_subtasks > 0 else 0
    
    # Self-refine usage rate: how often is self-refine triggered?
    subtasks_with_self_refine = sum(1 for s in stats['subtask_details'] if s['confirms'] > 0 or s['rejects'] > 0)
    metrics['self_refine_usage_rate'] = subtasks_with_self_refine / total_subtasks if total_subtasks > 0 else 0
    
    # Rejection patterns
    metrics['avg_rejects_per_subtask'] = stats['total_rejects'] / total_subtasks if total_subtasks > 0 else 0
    metrics['avg_confirms_per_subtask'] = stats['total_confirms'] / total_subtasks if total_subtasks > 0 else 0
    
    # Task-specific analysis
    task_type_stats = defaultdict(lambda: {'confirms': 0, 'rejects': 0, 'success': 0, 'total': 0})
    for subtask in stats['subtask_details']:
        task_type = subtask['task_type']
        task_type_stats[task_type]['total'] += 1
        task_type_stats[task_type]['confirms'] += subtask['confirms']
        task_type_stats[task_type]['rejects'] += subtask['rejects']
        if subtask['final_success']:
            task_type_stats[task_type]['success'] += 1
    
    metrics['task_type_breakdown'] = {}
    for task_type, stats_dict in task_type_stats.items():
        metrics['task_type_breakdown'][task_type] = {
            'success_rate': stats_dict['success'] / stats_dict['total'],
            'avg_confirms': stats_dict['confirms'] / stats_dict['total'],
            'avg_rejects': stats_dict['rejects'] / stats_dict['total'],
            'total_subtasks': stats_dict['total']
        }
    
    return metrics


def print_results(stats, metrics):
    """
    Print formatted results of the self-refine precision analysis.
    """
    
    print("=" * 60)
    print("SELF-REFINE PRECISION ANALYSIS")
    print("=" * 60)
    
    print(f"\n📊 OVERALL STATISTICS:")
    print(f"Total subtasks analyzed: {len(stats['subtask_details'])}")
    print(f"Total CONFIRM decisions: {stats['total_confirms']}")
    print(f"Total REJECT decisions: {stats['total_rejects']}")
    print(f"Subtasks using self-refine: {sum(1 for s in stats['subtask_details'] if s['confirms'] > 0 or s['rejects'] > 0)}")
    
    print(f"\n🎯 KEY METRICS:")
    if metrics['self_refine_precision'] is not None:
        print(f"Self-refine Precision (CONFIRM accuracy): {metrics['self_refine_precision']:.3f}")
    else:
        print(f"Self-refine Precision: N/A (no CONFIRM decisions)")
    
    print(f"Overall Success Rate: {metrics['overall_success_rate']:.3f}")
    print(f"Self-refine Usage Rate: {metrics['self_refine_usage_rate']:.3f}")
    print(f"Average Rejects per Subtask: {metrics['avg_rejects_per_subtask']:.2f}")
    print(f"Average Confirms per Subtask: {metrics['avg_confirms_per_subtask']:.2f}")
    
    print(f"\n📋 CONFIRM/REJECT BREAKDOWN:")
    print(f"CONFIRM → Success: {stats['confirm_success']}")
    print(f"CONFIRM → Failure: {stats['confirm_fail']}")
    print(f"Subtasks with Rejects → Success: {stats['reject_leading_to_success']}")
    print(f"Subtasks with Rejects → Failure: {stats['reject_leading_to_fail']}")
    
    print(f"\n📈 TASK TYPE BREAKDOWN:")
    for task_type, task_metrics in metrics['task_type_breakdown'].items():
        print(f"{task_type.upper()}:")
        print(f"  Success Rate: {task_metrics['success_rate']:.3f}")
        print(f"  Avg Confirms: {task_metrics['avg_confirms']:.2f}")
        print(f"  Avg Rejects: {task_metrics['avg_rejects']:.2f}")
        print(f"  Total Subtasks: {task_metrics['total_subtasks']}")
    
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Calculate self-refine precision from log files')
    parser.add_argument('log_file', help='Path to the log file to analyze')
    parser.add_argument('--output', '-o', help='Output JSON file for detailed results')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed subtask information')
    
    args = parser.parse_args()
    
    # Parse log file
    print(f"Analyzing log file: {args.log_file}")
    stats = parse_log_file(args.log_file)
    
    # Calculate metrics
    metrics = calculate_precision_metrics(stats)
    
    # Print results
    print_results(stats, metrics)
    
    # Show detailed subtask info if requested
    if args.verbose:
        print(f"\n🔍 DETAILED SUBTASK BREAKDOWN:")
        for subtask in stats['subtask_details']:
            print(f"Subtask {subtask['subtask_id']} ({subtask['task_type']}):")
            print(f"  Confirms: {subtask['confirms']}, Rejects: {subtask['rejects']}")
            print(f"  Final Decision: {subtask['final_decision']}")
            print(f"  Success: {subtask['final_success']}")
            print(f"  Choice Type: {subtask['choice_type']}")
    
    # Save to JSON if requested
    if args.output:
        results = {
            'statistics': stats,
            'metrics': metrics,
            'summary': {
                'log_file': args.log_file,
                'total_subtasks': len(stats['subtask_details']),
                'self_refine_precision': metrics['self_refine_precision'],
                'overall_success_rate': metrics['overall_success_rate']
            }
        }
        
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
