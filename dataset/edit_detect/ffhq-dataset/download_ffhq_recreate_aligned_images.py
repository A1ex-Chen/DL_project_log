def recreate_aligned_images(json_data, source_dir, dst_dir=
    'realign1024x1024', output_size=1024, transform_size=4096,
    enable_padding=True, rotate_level=True, random_shift=0.0, retry_crops=False
    ):
    print('Recreating aligned images...')
    np.random.seed(12345)
    _ = np.random.normal(0, 1, (len(json_data.values()), 2))
    if dst_dir:
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copyfile('LICENSE.txt', os.path.join(dst_dir, 'LICENSE.txt'))
    for item_idx, item in enumerate(json_data.values()):
        print('\r%d / %d ... ' % (item_idx, len(json_data)), end='', flush=True
            )
        lm = np.array(item['in_the_wild']['face_landmarks'])
        lm_chin = lm[0:17]
        lm_eyebrow_left = lm[17:22]
        lm_eyebrow_right = lm[22:27]
        lm_nose = lm[27:31]
        lm_nostrils = lm[31:36]
        lm_eye_left = lm[36:42]
        lm_eye_right = lm[42:48]
        lm_mouth_outer = lm[48:60]
        lm_mouth_inner = lm[60:68]
        eye_left = np.mean(lm_eye_left, axis=0)
        eye_right = np.mean(lm_eye_right, axis=0)
        eye_avg = (eye_left + eye_right) * 0.5
        eye_to_eye = eye_right - eye_left
        mouth_left = lm_mouth_outer[0]
        mouth_right = lm_mouth_outer[6]
        mouth_avg = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg
        if rotate_level:
            x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
            x /= np.hypot(*x)
            x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8
                )
            y = np.flipud(x) * [-1, 1]
            c0 = eye_avg + eye_to_mouth * 0.1
        else:
            x = np.array([1, 0], dtype=np.float64)
            x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8
                )
            y = np.flipud(x) * [-1, 1]
            c0 = eye_avg + eye_to_mouth * 0.1
        src_file = os.path.join(source_dir, item['in_the_wild']['file_path'])
        if not os.path.isfile(src_file):
            print(
                '\nCannot find source image. Please run "--wilds" before "--align".'
                )
            return
        img = PIL.Image.open(src_file)
        quad = np.stack([c0 - x - y, c0 - x + y, c0 + x + y, c0 + x - y])
        qsize = np.hypot(*x) * 2
        if random_shift != 0:
            for _ in range(1000):
                c = c0 + np.hypot(*x) * 2 * random_shift * np.random.normal(
                    0, 1, c0.shape)
                quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
                crop = int(np.floor(min(quad[:, 0]))), int(np.floor(min(
                    quad[:, 1]))), int(np.ceil(max(quad[:, 0]))), int(np.
                    ceil(max(quad[:, 1])))
                if not retry_crops or not (crop[0] < 0 or crop[1] < 0 or 
                    crop[2] >= img.width or crop[3] >= img.height):
                    break
            else:
                print('rejected image')
                return
        shrink = int(np.floor(qsize / output_size * 0.5))
        if shrink > 1:
            rsize = int(np.rint(float(img.size[0]) / shrink)), int(np.rint(
                float(img.size[1]) / shrink))
            img = img.resize(rsize, PIL.Image.ANTIALIAS)
            quad /= shrink
            qsize /= shrink
        border = max(int(np.rint(qsize * 0.1)), 3)
        crop = int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))
            ), int(np.ceil(max(quad[:, 0]))), int(np.ceil(max(quad[:, 1])))
        crop = max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop
            [2] + border, img.size[0]), min(crop[3] + border, img.size[1])
        if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
            img = img.crop(crop)
            quad -= crop[0:2]
        pad = int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))
            ), int(np.ceil(max(quad[:, 0]))), int(np.ceil(max(quad[:, 1])))
        pad = max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2
            ] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0)
        if enable_padding and max(pad) > border - 4:
            pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
            img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2
                ]), (0, 0)), 'reflect')
            h, w, _ = img.shape
            y, x, _ = np.ogrid[:h, :w, :1]
            mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.
                float32(w - 1 - x) / pad[2]), 1.0 - np.minimum(np.float32(y
                ) / pad[1], np.float32(h - 1 - y) / pad[3]))
            blur = qsize * 0.02
            img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img
                ) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
            img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0
                )
            img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255
                )), 'RGB')
            quad += pad[:2]
        img = img.transform((transform_size, transform_size), PIL.Image.
            QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
        if output_size < transform_size:
            img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)
        dst_subdir = os.path.join(dst_dir, '%05d' % (item_idx - item_idx % 
            1000))
        os.makedirs(dst_subdir, exist_ok=True)
        img.save(os.path.join(dst_subdir, '%05d.png' % item_idx))
    print('\r%d / %d ... done' % (len(json_data), len(json_data)))
