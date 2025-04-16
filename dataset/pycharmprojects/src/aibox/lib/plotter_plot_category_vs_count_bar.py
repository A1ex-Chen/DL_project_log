@staticmethod
def plot_category_vs_count_bar(category_vs_count_dict: Dict[str, int]):
    categories = [k for k in category_vs_count_dict.keys()]
    counts = [v for v in category_vs_count_dict.values()]
    category_and_count_list = [(category, count) for count, category in
        sorted(zip(counts, categories), reverse=True)]
    ax = sns.barplot(x=[category for category, _ in category_and_count_list
        ], y=[count for _, count in category_and_count_list])
    for patch in ax.patches:
        ax.annotate(f'{int(patch.get_height())}', (patch.get_x() + patch.
            get_width() / 2, patch.get_height()), ha='center', va='center',
            xytext=(0, 5), textcoords='offset points')
    ax.set_xlabel('Category')
    ax.set_ylabel('Count')
    fig = ax.get_figure()
    fig.tight_layout()
    plt.show()
    plt.close()
