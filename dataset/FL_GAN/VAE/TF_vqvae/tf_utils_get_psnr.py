def get_psnr(real, generated):
    psnr_value = tf.reduce_mean(tf.image.psnr(generated, real, max_val=255.0))
    return psnr_value
