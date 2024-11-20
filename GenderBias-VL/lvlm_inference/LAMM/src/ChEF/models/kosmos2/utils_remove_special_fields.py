def remove_special_fields(text):
    return re.sub('<.*?>', '', text)
