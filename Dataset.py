import tensorflow as tf

input = tf.io.read_file("/Users/anu/PycharmProjects/TensorFlow2/input.txt")
length = int(tf.strings.length(input))

vocab = tf.strings.unicode_split_with_offsets(input, 'UTF-8')
elem,idx = tf.unique(vocab[0])
vocab_size = len(elem)
print(f'Size of vocabulary={vocab_size}')
block_size = 32
table = tf.lookup.StaticHashTable(
    initializer=tf.lookup.KeyValueTensorInitializer(
        keys=elem,
        values=tf.constant([idx  for idx, inp in enumerate(elem)]),

    ),
    default_value=tf.constant(-1),
    name="elemtoindex"
)

indextoelem = tf.lookup.StaticHashTable(
    initializer=tf.lookup.KeyValueTensorInitializer(
        keys=tf.strings.as_string([idx  for idx, inp in enumerate(elem)]),
        values=elem,

    ),
    default_value=tf.constant('-1'),
    name="indextoelem"
)


global samplelist,reversesamplelist
samplelist = []
reversesamplelist = []

def reverse_map_fn(bytes):
    reversesamplelist.append(indextoelem.lookup(tf.strings.as_string(bytes)))
    return bytes

def map_fn(bytes):
    samplelist.append(table.lookup(bytes))
    return bytes

def random_sample(text,block_size):
    rand = tf.random.uniform(shape=[], minval=1, maxval=length - (block_size + 1))
    start = int(rand)
    # print(f'Start={int(rand)} Length={length} End={start + block_size + 1}')
    return tf.strings.substr(text,start, block_size, unit='BYTE')

def draw_random_sample(block_size):
        sample = random_sample(input,block_size)
        split_sample = tf.strings.bytes_split(sample)
        tf.map_fn(map_fn, tf.strings.bytes_split(split_sample))
        global samplelist
        reverse_map(samplelist[:-1])
        X,y = (tf.stack(samplelist[:-1]),tf.stack(samplelist[1:]))
        samplelist = []
        return X,y

def reverse_map(X):
    tf.map_fn(reverse_map_fn, X)

def decode(idx):
    return idx,indextoelem.lookup(
                    tf.strings.as_string([inp  for inp, inp in enumerate(idx)]))

