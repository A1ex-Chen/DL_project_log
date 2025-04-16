def extra_repr(self):
    st = (
        f'type={self.pos_type}, scale={self.scale}, normalize={self.normalize}'
        )
    if hasattr(self, 'gauss_B'):
        st += (
            f', gaussB={self.gauss_B.shape}, gaussBsum={self.gauss_B.sum().item()}'
            )
    return st
