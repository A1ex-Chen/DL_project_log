def encode_object_id(pid, tid):
    """
	Given process id (pid) and thread id (tid), return the object id.
	object id = pid (little endian 4 bytes) + tid (little endian 8 bytes)
	"""
    objId = struct.pack('<i', pid) + struct.pack('<q', tid)
    objId = binascii.hexlify(objId).decode('ascii').upper()
    return objId
