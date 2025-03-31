#!/usr/bin/env python3

import argparse
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main():
    parser = argparse.ArgumentParser(description='Visualize matrix multiplication benchmark results')
    parser.add_argument('-i', '--input', required=True,
                        help='Path to the CSV file with benchmark data')
    parser.add_argument('-o', '--output', default=None,
                        help='Base filename for output (default: derived from input filename)')
    args = parser.parse_args()

    input_filename = args.input

    if args.output:
        output_base = args.output
    else:
        output_base = os.path.splitext(os.path.basename(input_filename))[0]

    if not os.path.isfile(input_filename):
        print(f"Error: File '{input_filename}' not found.")
        return 1

    plt.style.use('ggplot')
    sns.set_theme(style="whitegrid")

    df = pd.read_csv(input_filename)

    df['duration_ms'] = df['duration_ns'] / 1_000_000

    plt.figure(figsize=(12, 9))

    implementations = {
        'Basic':      ['naive',     'trans_naive'],
        'Cache-line': ['cacheline', 'trans_cacheline'],
        'SIMD (SSE)': ['sse',       'trans_sse'],
        'SIMD (AVX)': ['avx',       'trans_avx']
    }

    colors = {
        'Basic':      ['#e41a1c', '#984ea3'],
        'Cache-line': ['#4daf4a', '#ff7f00'],
        'SIMD (SSE)': ['#377eb8', '#a65628'],
        'SIMD (AVX)': ['#f781bf', '#999999'],
        'Other':      ['#333333', '#777777']
    }

    markers = {
        'naive':           'o', 'trans_naive':     's',
        'cacheline':       '^', 'trans_cacheline': 'D',
        'sse':             'P', 'trans_sse':       'X',
        'avx':             '*', 'trans_avx':       'h',
    }

    linestyles = {
        'naive':           '-', 'trans_naive':     '--',
        'cacheline':       '-', 'trans_cacheline': '--',
        'sse':             '-', 'trans_sse':       '--',
        'avx':             '-', 'trans_avx':       '--',
    }

    all_implementations = df['implementation'].unique()

    for impl in all_implementations:
        if impl not in markers:
            markers[impl] = 'o'
        if impl not in linestyles:
            linestyles[impl] = '-'

    for group_idx, (group_name, impls) in enumerate(implementations.items()):
        for i, impl in enumerate(impls):
            impl_data = df[df['implementation'] == impl]
            if len(impl_data) > 0:
                plt.plot(
                    impl_data['matrix_size'],
                    impl_data['duration_ms'],
                    label=impl,
                    marker=markers[impl],
                    linestyle=linestyles[impl],
                    color=colors[group_name][i],
                    linewidth=2.5,
                    markersize=8
                )

    other_impls = [impl for impl in all_implementations if not any(impl in impls for _, impls in implementations.items())]
    for i, impl in enumerate(other_impls):
        impl_data = df[df['implementation'] == impl]
        plt.plot(
            impl_data['matrix_size'],
            impl_data['duration_ms'],
            label=impl,
            marker=markers[impl],
            linestyle=linestyles[impl],
            color=colors['Other'][i % len(colors['Other'])],
            linewidth=2.5,
            markersize=8
        )

    plt.title('Matrix Multiplication Performance Comparison', fontsize=16, fontweight='bold')
    plt.xlabel('Matrix Size (N×N×N)', fontsize=14)
    plt.ylabel('Execution Time (ms)', fontsize=14)

    plt.xscale('log', base=2)
    plt.yscale('log')

    plt.grid(True, which="both", ls="-", alpha=0.2)

    sizes = sorted(df['matrix_size'].unique())
    for size in sizes:
        plt.axvline(x=size, color='gray', linestyle=':', alpha=0.3)

    plt.xticks(sizes, [str(s) for s in sizes], rotation=45)

    legend_elements = []
    from matplotlib.lines import Line2D

    for group_name, impls in implementations.items():
        for i, impl in enumerate(impls):
            if impl in df['implementation'].unique():
                legend_elements.append(
                    Line2D([0], [0], color=colors[group_name][i], marker=markers[impl],
                           linestyle=linestyles[impl], linewidth=2.5, markersize=8, label=impl)
                )

    for i, impl in enumerate(other_impls):
        legend_elements.append(
            Line2D([0], [0], color=colors['Other'][i % len(colors['Other'])], marker=markers[impl],
                   linestyle=linestyles[impl], linewidth=2.5, markersize=8, label=impl)
        )

    legend = plt.legend(handles=legend_elements, title="Implementation",
                        loc="upper left", bbox_to_anchor=(1, 1),
                        fontsize=10, frameon=True)
    legend.get_title().set_fontsize('12')

    for impl in df['implementation'].unique():
        impl_data = df[df['implementation'] == impl]
        largest_size = impl_data['matrix_size'].max()
        last_duration = impl_data[impl_data['matrix_size'] == largest_size]['duration_ms'].values[0]

        plt.annotate(
            impl,
            xy=(largest_size, last_duration),
            xytext=(5, 0),
            textcoords="offset points",
            fontsize=8,
            color='black',
            va='center'
        )

    if 'naive' in df['implementation'].unique():
        naive_data = df[df['implementation'] == 'naive'].set_index('matrix_size')
        max_size = df['matrix_size'].max()
        fastest_impl = df[df['matrix_size'] == max_size].sort_values('duration_ms').iloc[0]['implementation']
        speedup = naive_data.loc[max_size, 'duration_ms'] / df[(df['implementation'] == fastest_impl) &
                                                               (df['matrix_size'] == max_size)]['duration_ms'].values[0]

        annotation_text = f"Maximum speedup: {speedup:.2f}x\n({fastest_impl} vs. naive at {max_size}×{max_size}×{max_size})"
        plt.annotate(
            annotation_text,
            xy=(0.02, 0.02),
            xycoords='figure fraction',
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8),
            fontsize=10
        )

    plt.tight_layout()
    plt.subplots_adjust(right=0.85)
    plt.savefig(f'{output_base}-performance.png', dpi=300, bbox_inches='tight')

    if 'naive' in df['implementation'].unique():
        plt.figure(figsize=(12, 6))

        for impl in df['implementation'].unique():
            if impl != 'naive':
                speedup_data = []
                sizes = sorted(df['matrix_size'].unique())

                for size in sizes:
                    naive_time = df[(df['implementation'] == 'naive') & (df['matrix_size'] == size)]['duration_ms'].values[0]
                    impl_time = df[(df['implementation'] == impl) & (df['matrix_size'] == size)]['duration_ms'].values[0]
                    speedup = naive_time / impl_time
                    speedup_data.append({'matrix_size': size, 'speedup': speedup})

                speedup_df = pd.DataFrame(speedup_data)

                impl_color = None
                impl_found = False

                for group_name, impls in implementations.items():
                    if impl in impls:
                        impl_idx = impls.index(impl)
                        impl_color = colors[group_name][impl_idx]
                        impl_found = True
                        break

                if not impl_found:
                    other_idx = other_impls.index(impl)
                    impl_color = colors['Other'][other_idx % len(colors['Other'])]

                plt.plot(
                    speedup_df['matrix_size'],
                    speedup_df['speedup'],
                    label=impl,
                    marker=markers[impl],
                    linestyle=linestyles[impl],
                    color=impl_color,
                    linewidth=2.5,
                    markersize=8
                )

        plt.title('Speedup Relative to Naive Implementation', fontsize=16, fontweight='bold')
        plt.xlabel('Matrix Size (N×N×N)', fontsize=14)
        plt.ylabel('Speedup Factor (×)', fontsize=14)

        plt.xscale('log', base=2)

        plt.grid(True, which="both", ls="-", alpha=0.2)
        sizes = sorted(df['matrix_size'].unique())
        for size in sizes:
            plt.axvline(x=size, color='gray', linestyle=':', alpha=0.3)

        plt.xticks(sizes, [str(s) for s in sizes], rotation=45)

        legend = plt.legend(title="Implementation", loc="upper left", bbox_to_anchor=(1, 1),
                            fontsize=10, frameon=True)
        legend.get_title().set_fontsize('12')

        plt.tight_layout()
        plt.subplots_adjust(right=0.85)
        plt.savefig(f'{output_base}-speedup.png', dpi=300, bbox_inches='tight')

    return 0

if __name__ == "__main__":
    sys.exit(main())