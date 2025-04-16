def request_with_credentials(url: str) ->any:
    """
    Make an AJAX request with cookies attached in a Google Colab environment.

    Args:
        url (str): The URL to make the request to.

    Returns:
        (any): The response data from the AJAX request.

    Raises:
        OSError: If the function is not run in a Google Colab environment.
    """
    if not IS_COLAB:
        raise OSError(
            'request_with_credentials() must run in a Colab environment')
    from google.colab import output
    from IPython import display
    display.display(display.Javascript(
        """
            window._hub_tmp = new Promise((resolve, reject) => {
                const timeout = setTimeout(() => reject("Failed authenticating existing browser session"), 5000)
                fetch("%s", {
                    method: 'POST',
                    credentials: 'include'
                })
                    .then((response) => resolve(response.json()))
                    .then((json) => {
                    clearTimeout(timeout);
                    }).catch((err) => {
                    clearTimeout(timeout);
                    reject(err);
                });
            });
            """
         % url))
    return output.eval_js('_hub_tmp')
