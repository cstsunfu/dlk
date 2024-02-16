# Copyright the author(s) of DLK.
# Copyright spark-nlp-display.
#
# This original source code is copied from spark-nlp-display and licensed under the Apache license.

from intc import Base, DictField, IntField, StrField, dataclass


@dataclass
class SpanAnnotation(Base):
    annotatorType = StrField(
        value="",
        help="The type of the output of the annotator. Possible values are ``DOCUMENT, TOKEN, WORDPIECE, WORD_EMBEDDINGS, SENTENCE_EMBEDDINGS, CATEGORY, DATE, ENTITY, SENTIMENT, POS, CHUNK, NAMED_ENTITY, NEGEX, DEPENDENCY, LABELED_DEPENDENCY, LANGUAGE, KEYWORD, DUMMY``.",
    )
    begin = IntField(
        value=0, help="The index of the first character under this annotation."
    )
    end = IntField(
        value=0, help="The index of the last character under this annotation."
    )
    result = StrField(value="", help="The resulting string of the annotation.")
    metadata = DictField(value={}, help="Associated metadata for this annotation")
