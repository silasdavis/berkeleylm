package edu.berkeley.nlp.lm.io;

import java.io.*;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.zip.GZIPInputStream;

import edu.berkeley.nlp.lm.NgramLanguageModel;
import edu.berkeley.nlp.lm.collections.Iterators;
import edu.berkeley.nlp.lm.util.Logger;
import edu.berkeley.nlp.lm.collections.Counter;

/**
 * Computes the log probability of a list of files. With the <code>-g</code>
 * option, it interprets the next two arguments as a <code>vocab_cs.gz</code>
 * file (see {@link LmReaders} for more detail) and a Berkeley LM binary,
 * respectively. Without <code>-g</code>, it interprets the next file as a
 * Berkeley LM binary. All remaining files are treated as plain-text (possibly
 * gzipped) files which have one sentence per line; a dash is used to indicate
 * that text should from standard input. If no files are given, reads from
 * standard input.
 *
 * @author adampauls
 */
public class ComputeNextWordsOfTextStream {

    private static void usage() {
        System.err.println("Usage: <Berkeley LM binary file> <input_file>*");
        System.exit(1);
    }

    public static void main(final String[] argv) throws IOException {
        int i = 0;
        String binaryFile = argv[i++];
        List<String> files = Arrays.asList(Arrays.copyOfRange(argv, i, argv.length));
        if (files.isEmpty()) files = Collections.singletonList("-");
        Logger.setGlobalLogger(new Logger.SystemLogger(System.out, System.err));
        NgramLanguageModel<String> lm = readBinary(binaryFile);
        computeNextProbs(files, lm);
    }

    /**
     * @param files
     * @param lm
     * @throws IOException
     */
    private static void computeNextProbs(List<String> files, NgramLanguageModel<String> lm) throws IOException {
        for (String file : files) {
            final InputStream is = (file.equals("-")) ? System.in : (file.endsWith(".gz") ? new GZIPInputStream(new FileInputStream(file))
                    : new FileInputStream(file));
            BufferedReader reader = new BufferedReader(new InputStreamReader(new BufferedInputStream(is)));
            for (String line : Iterators.able(IOUtils.lineIterator(reader))) {
                List<String> words = Arrays.asList(line.trim().split("\\s+")).subList(0, 2);
                Counter<String> c = NgramLanguageModel.StaticMethods.getDistributionOverNextWords(lm, words);
                System.out.println(c.toString());
            }
            Logger.endTrack();
        }
    }

    /**
     * @param binaryFile
     * @return
     */
    private static NgramLanguageModel<String> readBinary(String binaryFile) {
        NgramLanguageModel<String> lm;
        Logger.startTrack("Reading LM Binary " + binaryFile);
        lm = LmReaders.readLmBinary(binaryFile);
        Logger.endTrack();
        return lm;
    }

}
