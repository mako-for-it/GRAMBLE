### Wiktionary
i used the following wiktionary dump
enwiktionary-latest-pages-articles.xml.bz2
01-Dec-2025 23:47

My code should work on dumps from any timeline, but my work and evaluation is based on this dump.

i used this repository to extract from the dump
https://github.com/tatuylonen/wikitextprocessor

### Installations

# run
./run_lang.sh es 30000
./run_lang.sh nl 30000
./run_lang.sh af 30000
./run_lang.sh xh 30000
./run_lang.sh as 30000

./run_lang.sh es 30000 && ./run_lang.sh af 30000 && ./run_lang.sh xh 30000 && ./run_lang.sh as 30000

#eval

./run_en.sh es 30000 3000
./run_en.sh af 30000 3000
./run_en.sh xh 30000 3000
./run_en.sh nl 30000 3000

python3 train_translate_from_mlm.py --pair es-en --tgt es --samples 100
python3 train_translate_from_mlm.py --pair af-en --tgt af --samples 100
python3 train_translate_from_mlm.py --pair xh-en --tgt xh --samples 100
python3 train_translate_from_mlm.py --pair nl-en --tgt es --samples 100

./run_en.sh xh 30000 3000 \
&& ./run_en.sh nl 30000 3000 \
&& python3 train_translate_from_mlm.py --pair xh-en --tgt xh --samples 100 \
&& python3 train_translate_from_mlm.py --pair nl-en --tgt nl --samples 100