def sql_query(self, query: str, return_type: str='pandas') ->Union[Any, None]:
    """
        Run a SQL-Like query on the table. Utilizes LanceDB predicate pushdown.

        Args:
            query (str): SQL query to run.
            return_type (str): Type of the result to return. Can be either 'pandas' or 'arrow'. Defaults to 'pandas'.

        Returns:
            (pyarrow.Table): An arrow table containing the results.

        Example:
            ```python
            exp = Explorer()
            exp.create_embeddings_table()
            query = "SELECT * FROM 'table' WHERE labels LIKE '%person%'"
            result = exp.sql_query(query)
            ```
        """
    assert return_type in {'pandas', 'arrow'
        }, f'Return type should be either `pandas` or `arrow`, but got {return_type}'
    import duckdb
    if self.table is None:
        raise ValueError('Table is not created. Please create the table first.'
            )
    table = self.table.to_arrow()
    if not query.startswith('SELECT') and not query.startswith('WHERE'):
        raise ValueError(
            f'Query must start with SELECT or WHERE. You can either pass the entire query or just the WHERE clause. found {query}'
            )
    if query.startswith('WHERE'):
        query = f"SELECT * FROM 'table' {query}"
    LOGGER.info(f'Running query: {query}')
    rs = duckdb.sql(query)
    if return_type == 'arrow':
        return rs.arrow()
    elif return_type == 'pandas':
        return rs.df()
