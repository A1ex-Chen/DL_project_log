def plot_sql_query(self, query: str, labels: bool=True) ->Image.Image:
    """
        Plot the results of a SQL-Like query on the table.
        Args:
            query (str): SQL query to run.
            labels (bool): Whether to plot the labels or not.

        Returns:
            (PIL.Image): Image containing the plot.

        Example:
            ```python
            exp = Explorer()
            exp.create_embeddings_table()
            query = "SELECT * FROM 'table' WHERE labels LIKE '%person%'"
            result = exp.plot_sql_query(query)
            ```
        """
    result = self.sql_query(query, return_type='arrow')
    if len(result) == 0:
        LOGGER.info('No results found.')
        return None
    img = plot_query_result(result, plot_labels=labels)
    return Image.fromarray(img)
