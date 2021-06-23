class DummyDictionary(object):
    def __init__(
            self,
            size,
            pad='<pad>',
            bos='<s>'
    ):
        self.pad_word, self.bos_word = pad, bos
        self.size = size
        self.nspecial = 0

        self.pad_index = self.size
        self.nspecial += 1
        self.size += 1
        self.bos_index = self.size
        self.nspecial += 1
        self.size += 1

    def __len__(self):
        return self.size

    def bos(self):
        return self.bos_index

    def pad(self):
        return self.pad_index
