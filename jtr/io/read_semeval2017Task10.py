import os


def readAnn(textfolder="../data/SemEval2017Task10/"):
    '''
    Read .ann files and look up corresponding spans in .txt files
    
    Args:
        textfolder:
    '''

    flist = os.listdir(textfolder)
    for f in flist:
        if not f.endswith(".ann"):
            continue

        f_anno = open(os.path.join(textfolder, f), "rU")
        f_text = open(os.path.join(textfolder, f.replace(".ann", ".txt")), "rU")

        # there's only one line, as each .ann file is one text paragraph
        for l in f_text:
            text = l

        #@TODO: collect all keyphrase and relation annotations, create pairs of all keyphrase that appear in same sentence for USchema style RE

        for l in f_anno:
            anno_inst = l.strip().split("\t")
            if len(anno_inst) == 3:
                keytype, start, end = anno_inst[1].split(" ")
                if not keytype.endswith("-of"):

                    # look up span in text and print error message if it doesn't match the .ann span text
                    keyphr_text_lookup = text[int(start):int(end)]
                    keyphr_ann = anno_inst[2]
                    if keyphr_text_lookup != keyphr_ann:
                        print("Spans don't match for anno " + l.strip() + " in file " + f)

                #if keytype.endswith("-of"):


if __name__ == '__main__':
    readAnn()