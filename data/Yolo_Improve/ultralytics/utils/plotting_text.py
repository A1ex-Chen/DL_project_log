def text(self, xy, text, txt_color=(255, 255, 255), anchor='top', box_style
    =False):
    """Adds text to an image using PIL or cv2."""
    if anchor == 'bottom':
        w, h = self.font.getsize(text)
        xy[1] += 1 - h
    if self.pil:
        if box_style:
            w, h = self.font.getsize(text)
            self.draw.rectangle((xy[0], xy[1], xy[0] + w + 1, xy[1] + h + 1
                ), fill=txt_color)
            txt_color = 255, 255, 255
        if '\n' in text:
            lines = text.split('\n')
            _, h = self.font.getsize(text)
            for line in lines:
                self.draw.text(xy, line, fill=txt_color, font=self.font)
                xy[1] += h
        else:
            self.draw.text(xy, text, fill=txt_color, font=self.font)
    else:
        if box_style:
            w, h = cv2.getTextSize(text, 0, fontScale=self.sf, thickness=
                self.tf)[0]
            h += 3
            outside = xy[1] >= h
            p2 = xy[0] + w, xy[1] - h if outside else xy[1] + h
            cv2.rectangle(self.im, xy, p2, txt_color, -1, cv2.LINE_AA)
            txt_color = 255, 255, 255
        cv2.putText(self.im, text, xy, 0, self.sf, txt_color, thickness=
            self.tf, lineType=cv2.LINE_AA)
