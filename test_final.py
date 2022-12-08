from compressors.arithmetic_coding import AECParams, ArithmeticEncoder, ArithmeticDecoder
from compressors.probability_models import ContextTreeWeightKFreqModel
from core.prob_dist import Frequencies, ProbabilityDist

from typing import Tuple
from core.data_encoder_decoder import DataDecoder, DataEncoder
from core.data_block import DataBlock
import argparse
from utils.bitarray_utils import BitArray, uint_to_bitarray, bitarray_to_uint, float_to_bitarrays, bitarrays_to_float
from core.data_stream import Uint8FileDataStream, BitFileDataStream
from core.encoded_stream import EncodedBlockReader, EncodedBlockWriter
import pickle
import numpy as np

import cProfile
import gc

# constants
BLOCKSIZE = 10_000_000  # encode in 50 KB blocks

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--test", help="run tests", action="store_true")
parser.add_argument("-p", "--profile", help="run cProfile", action="store_true")
parser.add_argument("-d", "--decompress", help="decompress", action="store_true")
parser.add_argument("-i", "--input", help="input file", type=str)
parser.add_argument("-o", "--output", help="output file", type=str)


def test_sunehag():
    ctw = ContextTreeWeightKFreqModel([0, 1], 3, 1 << 61, [1, 1, 0])
    source = [0, 1, 0, 0, 1, 1, 0]
    for bit in source:
        ctw.update_model(bit) 
    print(np.exp2(ctw.root.lprob) * 2048)   
    ctw.update_model(0)
    print(np.exp2(ctw.root.lprob) * 65536)
    

class AECEmpiricalEncoder(DataEncoder):
    def encode_block(self, data_block: DataBlock):
        aec_params = AECParams()
        aec_encoder = ArithmeticEncoder(aec_params, ([0, 1], 48), ContextTreeWeightKFreqModel)
        # encode the data with Huffman code
        encoded_data = aec_encoder.encode_block(data_block)
        # return the Huffman encoding prepended with the encoded probability distribution
        return encoded_data

    def encode_file(self, input_file_path: str, encoded_file_path: str, block_size: int):
        """utility wrapper around the encode function using Uint8FileDataStream
        Args:
            input_file_path (str): path of the input file
            encoded_file_path (str): path of the encoded binary file
            block_size (int): choose the block size to be used to call the encode function
        """
        # call the encode function and write to the binary file
        with BitFileDataStream(input_file_path, "rb") as fds:
            with EncodedBlockWriter(encoded_file_path) as writer:
                self.encode(fds, block_size=block_size, encode_writer=writer)


class AECEmpiricalDecoder(DataDecoder):
    def decode_block(self, encoded_block: DataBlock):        
        aec_params = AECParams()
        aec_decoder = ArithmeticDecoder(aec_params, ([0, 1], 48), ContextTreeWeightKFreqModel)

        # now apply Huffman decoding
        decoded_data, num_bits_read = aec_decoder.decode_block(
            encoded_block
        )
        # verify we read all the bits provided
        assert num_bits_read == len(encoded_block)
        return decoded_data, len(encoded_block)

    def decode_file(self, encoded_file_path: str, output_file_path: str):
        """utility wrapper around the decode function using Uint8FileDataStream
        Args:
            encoded_file_path (str): input binary file
            output_file_path (str): output (text) file to which decoded data is written
        """
        # read from a binary file and decode data and write to a binary file
        with EncodedBlockReader(encoded_file_path) as reader:
            with BitFileDataStream(output_file_path, "wb") as fds:
                self.decode(reader, fds)


def main(args):
    tests = [ test_sunehag ]
    if args.test:
        for test_fn in tests:
            test_fn()
            
    elif args.decompress:
        decoder = AECEmpiricalDecoder()
        decoder.decode_file(args.input, args.output)
    else:
        encoder = AECEmpiricalEncoder()
        encoder.encode_file(args.input, args.output, block_size=BLOCKSIZE)
    

if __name__ == "__main__":
    args = parser.parse_args()
    if args.profile:
        cProfile.run('main(args)')
    else:
        main(args)
    
