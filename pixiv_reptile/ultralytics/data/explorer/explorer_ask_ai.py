def ask_ai(self, query):
    """
        Ask AI a question.

        Args:
            query (str): Question to ask.

        Returns:
            (pandas.DataFrame): A dataframe containing filtered results to the SQL query.

        Example:
            ```python
            exp = Explorer()
            exp.create_embeddings_table()
            answer = exp.ask_ai('Show images with 1 person and 2 dogs')
            ```
        """
    result = prompt_sql_query(query)
    try:
        return self.sql_query(result)
    except Exception as e:
        LOGGER.error(
            'AI generated query is not valid. Please try again with a different prompt'
            )
        LOGGER.error(e)
        return None
