import pandas as pd
import numpy as np
from collections import Counter
from itertools import islice

print("=" * 80)
print("WORKFLOW SEQUENCE PATTERN EXTRACTION")
print("=" * 80)

# Use df_sessions which has session_id already assigned
df_seq = df_sessions[['distinct_id', 'session_id', 'event', 'timestamp']].copy()
df_seq = df_seq.sort_values(['distinct_id', 'session_id', 'timestamp'])

# Extract n-grams (event sequences) for each user
def extract_ngrams(event_list, n):
    """Extract n-grams from event sequence"""
    if len(event_list) < n:
        return []
    return [tuple(event_list[i:i+n]) for i in range(len(event_list) - n + 1)]

print("\n✓ Extracting event sequences per user...")

# Group events by user
user_sequences = df_seq.groupby('distinct_id')['event'].apply(list).to_dict()

# Extract n-grams for n=3, 4, 5
user_ngrams = {}
for user_id, events in user_sequences.items():
    user_ngrams[user_id] = {
        'trigrams': extract_ngrams(events, 3),
        'fourgrams': extract_ngrams(events, 4),
        'fivegrams': extract_ngrams(events, 5)
    }

# Count most common sequences globally
all_trigrams = []
all_fourgrams = []
all_fivegrams = []

for user_id, ngrams in user_ngrams.items():
    all_trigrams.extend(ngrams['trigrams'])
    all_fourgrams.extend(ngrams['fourgrams'])
    all_fivegrams.extend(ngrams['fivegrams'])

trigram_counts = Counter(all_trigrams)
fourgram_counts = Counter(all_fourgrams)
fivegram_counts = Counter(all_fivegrams)

print(f"\n✓ Found {len(trigram_counts):,} unique 3-event sequences (trigrams)")
print(f"✓ Found {len(fourgram_counts):,} unique 4-event sequences (fourgrams)")
print(f"✓ Found {len(fivegram_counts):,} unique 5-event sequences (fivegrams)")

# Top sequences
top_trigrams = trigram_counts.most_common(20)
top_fourgrams = fourgram_counts.most_common(15)
top_fivegrams = fivegram_counts.most_common(10)

print("\n" + "=" * 80)
print("TOP 10 MOST COMMON 3-EVENT SEQUENCES (TRIGRAMS)")
print("=" * 80)
for i, (seq, count) in enumerate(top_trigrams[:10], 1):
    print(f"\n{i}. Count: {count:,}")
    print(f"   Sequence: {' → '.join(seq)}")

# Identify power user patterns: specific high-value sequences
power_user_patterns = [
    'block_create', 'block_run', 'block_execution_success',
    'agent_accept_suggestion', 'agent_worker_created', 'agent_worker_finished',
    'canvas_created', 'canvas_shared', 'api_deployed',
    'model_trained', 'model_deployed'
]

# Identify struggle patterns: repeated errors, abandoned workflows
struggle_indicators = [
    'block_execution_failed', 'error_occurred', 'exception_caught',
    'connection_failed', 'query_failed'
]

print("\n" + "=" * 80)
print("POWER USER & STRUGGLE PATTERN IDENTIFICATION")
print("=" * 80)

# Calculate per-user sequence features
user_sequence_features = []

for user_id, events in user_sequences.items():
    event_counts = Counter(events)
    
    # Power user indicators
    power_events_count = sum(event_counts.get(evt, 0) for evt in power_user_patterns if evt in event_counts)
    has_deployment_sequence = any('deploy' in str(evt).lower() for evt in events)
    has_agent_workflow = any('agent' in str(evt).lower() for evt in events)
    has_create_run_pattern = any(
        events[i:i+2] == ['block_create', 'block_run'] or
        events[i:i+2] == ['block_created', 'block_execution_started']
        for i in range(len(events)-1)
    )
    
    # Struggle indicators
    error_count = sum(event_counts.get(evt, 0) for evt in struggle_indicators if evt in event_counts)
    failed_block_runs = event_counts.get('block_execution_failed', 0)
    
    # Repetition analysis (same event multiple times in short succession)
    repeated_events = 0
    for i in range(len(events) - 2):
        if events[i] == events[i+1] == events[i+2]:
            repeated_events += 1
    
    # Sequence diversity
    unique_sequences = len(set(user_ngrams[user_id]['trigrams']))
    total_sequences = len(user_ngrams[user_id]['trigrams'])
    sequence_diversity = unique_sequences / max(total_sequences, 1)
    
    # Common power patterns
    trigram_set = set(user_ngrams[user_id]['trigrams'])
    has_create_edit_run = any(
        ('create' in str(t[0]).lower() and 'edit' in str(t[1]).lower() and 'run' in str(t[2]).lower())
        or ('create' in str(t[0]).lower() and 'run' in str(t[1]).lower())
        for t in trigram_set
    )
    
    user_sequence_features.append({
        'user_id': user_id,
        'total_event_count': len(events),
        'unique_event_types': len(event_counts),
        'power_events_count': power_events_count,
        'has_deployment_sequence': int(has_deployment_sequence),
        'has_agent_workflow': int(has_agent_workflow),
        'has_create_run_pattern': int(has_create_run_pattern),
        'error_count': error_count,
        'failed_block_runs': failed_block_runs,
        'repeated_events_count': repeated_events,
        'sequence_diversity': sequence_diversity,
        'has_create_edit_run': int(has_create_edit_run),
        'trigram_count': len(user_ngrams[user_id]['trigrams']),
        'fourgram_count': len(user_ngrams[user_id]['fourgrams']),
        'fivegram_count': len(user_ngrams[user_id]['fivegrams'])
    })

workflow_sequence_df = pd.DataFrame(user_sequence_features)

# Calculate composite scores
workflow_sequence_df['power_user_score'] = (
    workflow_sequence_df['power_events_count'] * 0.4 +
    workflow_sequence_df['has_deployment_sequence'] * 10 +
    workflow_sequence_df['has_agent_workflow'] * 5 +
    workflow_sequence_df['has_create_run_pattern'] * 8 +
    workflow_sequence_df['has_create_edit_run'] * 6
)

workflow_sequence_df['struggle_score'] = (
    workflow_sequence_df['error_count'] * 2 +
    workflow_sequence_df['failed_block_runs'] * 3 +
    workflow_sequence_df['repeated_events_count'] * 1.5
)

print(f"\n✓ Extracted workflow sequence features for {len(workflow_sequence_df):,} users")
print(f"\nPower user score range: {workflow_sequence_df['power_user_score'].min():.1f} to {workflow_sequence_df['power_user_score'].max():.1f}")
print(f"Struggle score range: {workflow_sequence_df['struggle_score'].min():.1f} to {workflow_sequence_df['struggle_score'].max():.1f}")

print("\n" + "=" * 80)
print("WORKFLOW SEQUENCE SUMMARY STATISTICS")
print("=" * 80)
print(workflow_sequence_df.describe().round(2).to_string())

print("\n✓ Workflow sequence pattern extraction complete!")