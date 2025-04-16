def text(self, xy, text, txt_color=(255, 255, 255)):
    w, h = self.font.getsize(text)
    self.draw.text((xy[0], xy[1] - h + 1), text, fill=txt_color, font=self.font
        )
