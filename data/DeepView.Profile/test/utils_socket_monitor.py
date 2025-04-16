def socket_monitor(socket, callback=None):
    try:
        while socket:
            ret = select.select([socket], [], [], 1)
            if len(ret[0]):
                msg_len = struct.unpack('>I', socket.recv(4))[0]
                buf = None
                while socket and msg_len:
                    msg = socket.recv(msg_len)
                    if len(msg) == 0:
                        break
                    buf = buf + msg if buf else msg
                    msg_len -= len(msg)
                callback(buf)
    except (ValueError, OSError):
        print(f'Closing listener for socket {socket}')
