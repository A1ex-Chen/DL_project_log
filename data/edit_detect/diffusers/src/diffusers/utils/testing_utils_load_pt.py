def load_pt(url: str):
    response = requests.get(url)
    response.raise_for_status()
    arry = torch.load(BytesIO(response.content))
    return arry
