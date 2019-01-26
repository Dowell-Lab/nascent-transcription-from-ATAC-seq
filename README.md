# nascent-transcription-from-ATAC-seq
Companion material to the paper "Detecting transcription from ATAC-seq]{ATAC-seq signal processing and recurrent neural networks can identify RNA polymerase activity"

This repository contains the saved machine learning models for both the recurrent neural network utilizing the hybrid nucleotide encoding, and the AdaBoost classifier from the signal-only encoding. Both have been trained on every ATAC-seq sample from the publication:

    SRR1822165
    SRR1822166
    SRR1822167
    SRR1822168
    SRR5007258
    SRR5007259
    SRR5876158
    SRR5876159
    SRR5128074

All scripts included are tailored to our supercomputing cluster environment, however they can easily be adapted to be trained or tested using any other samples. The dataset used is a pickled Pandas dataframe containing one row per ATAC-seq peak, a `sequence` field containing a string of nucleotides for the evaluated window, and a `signal_features` field containing an array of real values containing the normalized coverage over the evaluation window. Each dataframe row also contains the peak coordinates in the fields `chrom`, `start` and `end`. The RNN model implemented with Keras has been tested on the following NVidia GPUs: Titan, Tesla-K20, Tesla-K40, and Gtx 1080ti.
