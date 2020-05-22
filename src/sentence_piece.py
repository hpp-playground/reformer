# %%
import sentencepiece as spm
import codecs
import re
import requests
from bs4 import BeautifulSoup

# %%
url = "https://ja.wikipedia.org/wiki/"
keyword_list = [
    "織田信長", "徳川家康", "豊臣秀吉",
    "伊達政宗", "武田信玄", "上杉謙信",
    "明智光秀", "島津義弘", "北条氏康",
    "長宗我部元親", "毛利元就", "真田幸村",
    "立花宗茂", "石田三成", "浅井長政"
]

# %%
corpus = []
for keyword in keyword_list:
    # 戦国大名の記事をダウンロード
    response = requests.get(url + keyword)
    # htmlに変換
    html = response.text

    soup = BeautifulSoup(html, 'lxml')
    for p_tag in soup.find_all('p'):
        text = "".join(p_tag.text.strip().split(" "))
        if len(text) == 0:
            continue
        # e.g. [注釈9] -> ''
        text = re.sub(r"\[注釈[0-9]+\]", "", text)
        # e.g. [注9] -> ''
        text = re.sub(r"\[注[0-9]+\]", "", text)
        # e.g. [20] -> ''
        text = re.sub(r"\[[0-9]+\]", "", text)
        corpus.append(text)

print(*corpus, sep="\n", file=codecs.open("wiki-daimyo.txt", "w", "utf-8"))
# len(corpus) ~= 1500

# %%
spm.SentencePieceTrainer.Train(
    '--input=wiki-daimyo.txt, --model_prefix=sentencepiece --character_coverage=0.9995 --vocab_size=8000 --pad_id=3'
)

# %%
sp = spm.SentencePieceProcessor()
sp.Load("sentencepiece.model")

# %%
print(sp.EncodeAsPieces(corpus[70]))

# %%
print(sp.EncodeAsIds(corpus[70]))
# [16, 111, 17, 211, 28, 4, 4228, 67, 2268, 10, 1564, 20, 4, 6106, 3832, 81, 15, 2268, 124, 55, 2311, 23, 2268, 157, 12, 4909, 366, 476, 1050, 61, 15, 113, 7923, 242, 13, 7, 4, 3870, 15, 98, 1010, 13, 6069, 4, 6106, 653, 2377, 6, 52, 4, 3030, 997, 1144, 10, 6024, 18, 2738, 1537, 20, 4, 93, 1413, 5, 3947, 39, 10, 1564, 12, 654, 101, 2634, 63, 70, 7974, 232, 15, 4582, 55, 1195, 56, 4, 793, 2180, 12, 4388, 51, 958, 736, 529, 9, 4287, 10, 4, 76, 17, 5, 2015, 18, 229, 12, 2635, 7, 2738, 1537, 20, 4, 7784, 285, 7, 3539, 6]

# %%
tokens = ['松永久秀', 'らが', '共', '謀', 'して', '秀吉に', '敵対し', 'た', '。']
print(sp.DecodePieces(tokens))
# 松永久秀らが共謀して秀吉に敵対した。

# %%
ids = sp.EncodeAsIds(corpus[70])
print(ids)
# [16, 111, 17, 211, 28, 4, 4228, 67, 2268, 10, 1564, 20, 4, 6106, 3832, 81, 15, 2268, 124, 55, 2311, 23, 2268, 157, 12, 4909, 366, 476, 1050, 61, 15, 113, 7923, 242, 13, 7, 4, 3870, 15, 98, 1010, 13, 6069, 4, 6106, 653, 2377, 6, 52, 4, 3030, 997, 1144, 10, 6024, 18, 2738, 1537, 20, 4, 93, 1413, 5, 3947, 39, 10, 1564, 12, 654, 101, 2634, 63, 70, 7974, 232, 15, 4582, 55, 1195, 56, 4, 793, 2180, 12, 4388, 51, 958, 736, 529, 9, 4287, 10, 4, 76, 17, 5, 2015, 18, 229, 12, 2635, 7, 2738, 1537, 20, 4, 7784, 285, 7, 3539, 6]
print(sp.DecodeIds(ids))
# 11月14日、織田方であった岩村城が開城し、武田方に占拠された(岩村城の戦い)。病死した岩村城主・遠山景任の後家(信長の叔母)は、秋山虎繁(信友)と婚姻し、武田方に転じた。また、徳川領においては徳川軍が一言坂の戦いで武田軍に敗退し、さらに遠江国の二俣城が開城・降伏により不利な戦況となる(二俣城の戦い)。これに対して信長は、家康に佐久間信盛・平手汎秀ら3,000人の援軍を送ったが、12月の三方ヶ原の戦いで織田・徳川連合軍は武田軍に敗退し、汎秀は討死した。

# %%
print(sp.GetPieceSize())
# 8000

# %%
print(sp.PieceToId('</s>'))
print(sp["</s>"])

# %%
for i in sp.EncodeAsIds("織田信長が勝利した"):
    print(i, sp.IdToPiece(i))

# %%
for i in sp.EncodeAsIds("セネガル勝ってるよ"):
    print(i, sp.IdToPiece(i))

# %%
for i in sp.EncodeAsIds("本田圭佑勝ってるよ"):
    print(i, sp.IdToPiece(i))

# %%
