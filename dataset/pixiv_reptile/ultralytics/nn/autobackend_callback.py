def callback(request, userdata):
    """Places result in preallocated list using userdata index."""
    results[userdata] = request.results
