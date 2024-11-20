def _clean_caption(self, caption):
    caption = str(caption)
    caption = ul.unquote_plus(caption)
    caption = caption.strip().lower()
    caption = re.sub('<person>', 'person', caption)
    caption = re.sub(
        '\\b((?:https?:(?:\\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\\w/-]*\\b\\/?(?!@)))'
        , '', caption)
    caption = re.sub(
        '\\b((?:www:(?:\\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\\w/-]*\\b\\/?(?!@)))'
        , '', caption)
    caption = BeautifulSoup(caption, features='html.parser').text
    caption = re.sub('@[\\w\\d]+\\b', '', caption)
    caption = re.sub('[\\u31c0-\\u31ef]+', '', caption)
    caption = re.sub('[\\u31f0-\\u31ff]+', '', caption)
    caption = re.sub('[\\u3200-\\u32ff]+', '', caption)
    caption = re.sub('[\\u3300-\\u33ff]+', '', caption)
    caption = re.sub('[\\u3400-\\u4dbf]+', '', caption)
    caption = re.sub('[\\u4dc0-\\u4dff]+', '', caption)
    caption = re.sub('[\\u4e00-\\u9fff]+', '', caption)
    caption = re.sub(
        '[\\u002D\\u058A\\u05BE\\u1400\\u1806\\u2010-\\u2015\\u2E17\\u2E1A\\u2E3A\\u2E3B\\u2E40\\u301C\\u3030\\u30A0\\uFE31\\uFE32\\uFE58\\uFE63\\uFF0D]+'
        , '-', caption)
    caption = re.sub('[`´«»“”¨]', '"', caption)
    caption = re.sub('[‘’]', "'", caption)
    caption = re.sub('&quot;?', '', caption)
    caption = re.sub('&amp', '', caption)
    caption = re.sub('\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}', ' ', caption)
    caption = re.sub('\\d:\\d\\d\\s+$', '', caption)
    caption = re.sub('\\\\n', ' ', caption)
    caption = re.sub('#\\d{1,3}\\b', '', caption)
    caption = re.sub('#\\d{5,}\\b', '', caption)
    caption = re.sub('\\b\\d{6,}\\b', '', caption)
    caption = re.sub('[\\S]+\\.(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)',
        '', caption)
    caption = re.sub('[\\"\\\']{2,}', '"', caption)
    caption = re.sub('[\\.]{2,}', ' ', caption)
    caption = re.sub(self.bad_punct_regex, ' ', caption)
    caption = re.sub('\\s+\\.\\s+', ' ', caption)
    regex2 = re.compile('(?:\\-|\\_)')
    if len(re.findall(regex2, caption)) > 3:
        caption = re.sub(regex2, ' ', caption)
    caption = ftfy.fix_text(caption)
    caption = html.unescape(html.unescape(caption))
    caption = re.sub('\\b[a-zA-Z]{1,3}\\d{3,15}\\b', '', caption)
    caption = re.sub('\\b[a-zA-Z]+\\d+[a-zA-Z]+\\b', '', caption)
    caption = re.sub('\\b\\d+[a-zA-Z]+\\d+\\b', '', caption)
    caption = re.sub('(worldwide\\s+)?(free\\s+)?shipping', '', caption)
    caption = re.sub('(free\\s)?download(\\sfree)?', '', caption)
    caption = re.sub('\\bclick\\b\\s(?:for|on)\\s\\w+', '', caption)
    caption = re.sub(
        '\\b(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)(\\simage[s]?)?', '',
        caption)
    caption = re.sub('\\bpage\\s+\\d+\\b', '', caption)
    caption = re.sub('\\b\\d*[a-zA-Z]+\\d+[a-zA-Z]+\\d+[a-zA-Z\\d]*\\b',
        ' ', caption)
    caption = re.sub('\\b\\d+\\.?\\d*[xх×]\\d+\\.?\\d*\\b', '', caption)
    caption = re.sub('\\b\\s+\\:\\s+', ': ', caption)
    caption = re.sub('(\\D[,\\./])\\b', '\\1 ', caption)
    caption = re.sub('\\s+', ' ', caption)
    caption.strip()
    caption = re.sub('^[\\"\\\']([\\w\\W]+)[\\"\\\']$', '\\1', caption)
    caption = re.sub("^[\\'\\_,\\-\\:;]", '', caption)
    caption = re.sub("[\\'\\_,\\-\\:\\-\\+]$", '', caption)
    caption = re.sub('^\\.\\S+$', '', caption)
    return caption.strip()
