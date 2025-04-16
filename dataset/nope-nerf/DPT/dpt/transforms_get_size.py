def get_size(self, width, height):
    scale_height = self.__height / height
    scale_width = self.__width / width
    if self.__keep_aspect_ratio:
        if self.__resize_method == 'lower_bound':
            if scale_width > scale_height:
                scale_height = scale_width
            else:
                scale_width = scale_height
        elif self.__resize_method == 'upper_bound':
            if scale_width < scale_height:
                scale_height = scale_width
            else:
                scale_width = scale_height
        elif self.__resize_method == 'minimal':
            if abs(1 - scale_width) < abs(1 - scale_height):
                scale_height = scale_width
            else:
                scale_width = scale_height
        else:
            raise ValueError(
                f'resize_method {self.__resize_method} not implemented')
    if self.__resize_method == 'lower_bound':
        new_height = self.constrain_to_multiple_of(scale_height * height,
            min_val=self.__height)
        new_width = self.constrain_to_multiple_of(scale_width * width,
            min_val=self.__width)
    elif self.__resize_method == 'upper_bound':
        new_height = self.constrain_to_multiple_of(scale_height * height,
            max_val=self.__height)
        new_width = self.constrain_to_multiple_of(scale_width * width,
            max_val=self.__width)
    elif self.__resize_method == 'minimal':
        new_height = self.constrain_to_multiple_of(scale_height * height)
        new_width = self.constrain_to_multiple_of(scale_width * width)
    else:
        raise ValueError(
            f'resize_method {self.__resize_method} not implemented')
    return new_width, new_height
