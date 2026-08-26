"""Byte kernels used by the Python FASTA/FASTQ indexer."""

from std.sys.info import simd_width_of as simdwidthof

comptime BytePtr = Pointer[UInt8, AnyOrigin[mut=True]]
comptime IntPtr = Pointer[Int64, AnyOrigin[mut=True]]
comptime BYTE_W = simdwidthof[DType.uint8]()


def canonical_simd[W: Int](chars: SIMD[DType.uint8, W]) -> Bool:
    var lower = chars | SIMD[DType.uint8, W](32)
    var valid = (
        lower.eq(UInt8(97)) | lower.eq(UInt8(99)) | lower.eq(UInt8(103))
        | lower.eq(UInt8(110)) | lower.eq(UInt8(116)) | lower.eq(UInt8(117))
    )
    return Int(valid.select(SIMD[DType.uint8, W](1), SIMD[DType.uint8, W](0)).reduce_add()) == W


def canonical_complement_simd[W: Int](chars: SIMD[DType.uint8, W]) -> SIMD[DType.uint8, W]:
    var lower = chars | SIMD[DType.uint8, W](32)
    var result = chars
    result = lower.eq(UInt8(97)).select(SIMD[DType.uint8, W](116), result)
    result = lower.eq(UInt8(99)).select(SIMD[DType.uint8, W](103), result)
    result = lower.eq(UInt8(103)).select(SIMD[DType.uint8, W](99), result)
    result = lower.eq(UInt8(110)).select(SIMD[DType.uint8, W](110), result)
    result = (lower.eq(UInt8(116)) | lower.eq(UInt8(117))).select(SIMD[DType.uint8, W](97), result)
    var uppercase_mask = (chars & SIMD[DType.uint8, W](32)).eq(UInt8(0)).select(
        SIMD[DType.uint8, W](32), SIMD[DType.uint8, W](0)
    )
    return result ^ uppercase_mask


def complement_simd[W: Int](chars: SIMD[DType.uint8, W]) -> SIMD[DType.uint8, W]:
    var result = chars
    result = chars.eq(UInt8(65)).select(SIMD[DType.uint8, W](84), result)
    result = chars.eq(UInt8(67)).select(SIMD[DType.uint8, W](71), result)
    result = chars.eq(UInt8(71)).select(SIMD[DType.uint8, W](67), result)
    result = (chars.eq(UInt8(84)) | chars.eq(UInt8(85))).select(SIMD[DType.uint8, W](65), result)
    result = chars.eq(UInt8(97)).select(SIMD[DType.uint8, W](116), result)
    result = chars.eq(UInt8(99)).select(SIMD[DType.uint8, W](103), result)
    result = chars.eq(UInt8(103)).select(SIMD[DType.uint8, W](99), result)
    result = (chars.eq(UInt8(116)) | chars.eq(UInt8(117))).select(SIMD[DType.uint8, W](97), result)
    result = chars.eq(UInt8(77)).select(SIMD[DType.uint8, W](75), result)
    result = chars.eq(UInt8(82)).select(SIMD[DType.uint8, W](89), result)
    result = chars.eq(UInt8(89)).select(SIMD[DType.uint8, W](82), result)
    result = chars.eq(UInt8(75)).select(SIMD[DType.uint8, W](77), result)
    result = chars.eq(UInt8(86)).select(SIMD[DType.uint8, W](66), result)
    result = chars.eq(UInt8(72)).select(SIMD[DType.uint8, W](68), result)
    result = chars.eq(UInt8(68)).select(SIMD[DType.uint8, W](72), result)
    result = chars.eq(UInt8(66)).select(SIMD[DType.uint8, W](86), result)
    result = chars.eq(UInt8(109)).select(SIMD[DType.uint8, W](107), result)
    result = chars.eq(UInt8(114)).select(SIMD[DType.uint8, W](121), result)
    result = chars.eq(UInt8(121)).select(SIMD[DType.uint8, W](114), result)
    result = chars.eq(UInt8(107)).select(SIMD[DType.uint8, W](109), result)
    result = chars.eq(UInt8(118)).select(SIMD[DType.uint8, W](98), result)
    result = chars.eq(UInt8(104)).select(SIMD[DType.uint8, W](100), result)
    result = chars.eq(UInt8(100)).select(SIMD[DType.uint8, W](104), result)
    result = chars.eq(UInt8(98)).select(SIMD[DType.uint8, W](118), result)
    return result


def transform_range(src: BytePtr, dst: BytePtr, mapping: BytePtr, n: Int, mode: Int,
                    start: Int, end: Int):
    var i = start
    if mode == 0:
        while i + BYTE_W <= end:
            var chars = src.unsafe_load[width=BYTE_W, alignment=1](i)
            if canonical_simd[BYTE_W](chars):
                dst.unsafe_store(i, canonical_complement_simd[BYTE_W](chars))
            else:
                dst.unsafe_store(i, complement_simd[BYTE_W](chars))
            i += BYTE_W
        while i < end:
            dst[unsafe_offset=i] = mapping[
                unsafe_offset=Int(src[unsafe_offset=i])
            ]
            i += 1
    elif mode == 1:
        while i + BYTE_W <= end:
            dst.unsafe_store(
                i,
                src.unsafe_load[width=BYTE_W, alignment=1](n - i - BYTE_W).reversed(),
            )
            i += BYTE_W
        while i < end:
            dst[unsafe_offset=i] = src[unsafe_offset=n - 1 - i]
            i += 1
    else:
        while i + BYTE_W <= end:
            var chars = src.unsafe_load[width=BYTE_W, alignment=1](n - i - BYTE_W).reversed()
            if canonical_simd[BYTE_W](chars):
                dst.unsafe_store(i, canonical_complement_simd[BYTE_W](chars))
            else:
                dst.unsafe_store(i, complement_simd[BYTE_W](chars))
            i += BYTE_W
        while i < end:
            dst[unsafe_offset=i] = mapping[
                unsafe_offset=Int(src[unsafe_offset=n - 1 - i])
            ]
            i += 1


@export("mpf_transform")
def mpf_transform(src_addr: Int, dst_addr: Int, map_addr: Int, n: Int, mode: Int) abi("C"):
    """Apply complement, reverse, or reverse-complement mode."""
    # This is a C ABI: reject invalid calls before materializing an unsafe pointer.
    # Python validates these too, but the guard keeps an accidental direct caller
    # from dereferencing a null address or walking a negative range.
    if n <= 0 or src_addr == 0 or dst_addr == 0 or map_addr == 0 or mode < 0 or mode > 2:
        return
    var src = BytePtr(unsafe_from_address=src_addr)
    var dst = BytePtr(unsafe_from_address=dst_addr)
    var mapping = BytePtr(unsafe_from_address=map_addr)
    transform_range(src, dst, mapping, n, mode, 0, n)


@export("mpf_count_bytes")
def mpf_count_bytes(src_addr: Int, counts_addr: Int, n: Int) abi("C"):
    """Accumulate all 256 byte values in an Int64[256] histogram."""
    if n <= 0 or src_addr == 0 or counts_addr == 0:
        return
    var src = BytePtr(unsafe_from_address=src_addr)
    var counts = IntPtr(unsafe_from_address=counts_addr)
    for i in range(n):
        counts[unsafe_offset=Int(src[unsafe_offset=i])] += 1
