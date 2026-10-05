/*
 * kernels/_engine/kv_quant/generic/kv_quant_parity.c
 *
 * Emits the portable codec's bytes and decoded values for a fixed corpus, as
 * hex, so the Rust test in crates/spite-kvcache/tests/portable_codec_parity.rs
 * can compare them with `spite_kvcache::quant` bit for bit.
 *
 * Output (one line per tier, two lines per tier):
 *   E <tier> <hex of every encoded block>
 *   D <tier> <hex of every decoded f32 bit pattern>
 *
 * Every byte printed derives from the corpus below, which the Rust test
 * reproduces exactly, so a mismatch is a real disagreement about the layout.
 */

#include <stdio.h>

#include "kernels/_engine/kv_quant/generic/kv_quant_blocks.h"

#define KQP_BLOCKS 64
#define KQP_ELEMS  (KQP_BLOCKS * KVQ_BLOCK_ELEMS)

/*
 * Deterministic corpus shared with the Rust test. Half the blocks are scaled
 * tiny so their block scale lands in the f16 subnormal range, which is where
 * the codecs previously disagreed.
 */
static float kqp_corpus(int i) {
    uint32_t x = (uint32_t)i * 2654435761u;
    float scale, mag;

    x ^= x >> 15;
    x *= 2246822519u;
    x ^= x >> 13;

    scale = ((i / KVQ_BLOCK_ELEMS) % 2 == 0) ? 1e-3f : 1e-6f;
    mag = (float)(x % 8192u) * scale;
    return (x & 0x80000000u) ? -mag : mag;
}

typedef void (*kqp_enc_fn)(const float*, uint8_t*);
typedef void (*kqp_dec_fn)(const uint8_t*, float*);

static void kqp_emit(const char* tier, kqp_enc_fn enc, kqp_dec_fn dec, int block_bytes,
                     const float* src, uint8_t* buf, float* dec_out) {
    int b, i;

    for (b = 0; b < KQP_BLOCKS; ++b)
        enc(src + b * KVQ_BLOCK_ELEMS, buf + b * block_bytes);
    printf("E %s ", tier);
    for (i = 0; i < KQP_BLOCKS * block_bytes; ++i)
        printf("%02x", buf[i]);
    printf("\n");

    for (b = 0; b < KQP_BLOCKS; ++b)
        dec(buf + b * block_bytes, dec_out + b * KVQ_BLOCK_ELEMS);
    printf("D %s ", tier);
    for (i = 0; i < KQP_ELEMS; ++i)
        printf("%08x", kvq_f32_bits(dec_out[i]));
    printf("\n");
}

int main(void) {
    static float src[KQP_ELEMS];
    static uint8_t buf[KQP_BLOCKS * KVQ_Q8_0_BYTES];
    static float dec_out[KQP_ELEMS];
    int i;

    for (i = 0; i < KQP_ELEMS; ++i)
        src[i] = kqp_corpus(i);

    kqp_emit("q8_0", kvq_encode_q8_0, kvq_decode_q8_0, KVQ_Q8_0_BYTES, src, buf, dec_out);
    kqp_emit("q5_1", kvq_encode_q5_1, kvq_decode_q5_1, KVQ_Q5_1_BYTES, src, buf, dec_out);
    kqp_emit("q4_0", kvq_encode_q4_0, kvq_decode_q4_0, KVQ_Q4_0_BYTES, src, buf, dec_out);
    return 0;
}
