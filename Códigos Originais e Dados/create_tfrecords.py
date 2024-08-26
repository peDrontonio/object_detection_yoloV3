import tensorflow as tf
import os
import json

def create_tf_example(image_info, label_path):
    with open(label_path, 'r') as f:
        labels = f.readlines()

    image_path = image_info['file_name']
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image)

    height = image_info['height']
    width = image_info['width']

    xmins, xmaxs, ymins, ymaxs, classes, classes_text = [], [], [], [], [], []

    for label in labels:
        values = list(map(float, label.split()))
        x_center, y_center, w, h = values[1], values[2], values[3], values[4]

        xmins.append((x_center - w / 2) / width)
        xmaxs.append((x_center + w / 2) / width)
        ymins.append((y_center - h / 2) / height)
        ymaxs.append((y_center + h / 2) / height)
        classes.append(int(values[0]))
        classes_text.append(str(values[0]).encode('utf8'))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_path.encode('utf8')])),
        'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_path.encode('utf8')])),
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(image).numpy()])),
        'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=['jpeg'.encode('utf8')])),
        'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmins)),
        'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmaxs)),
        'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymins)),
        'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymaxs)),
        'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_text)),
        'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=classes)),
    }))
    return tf_example

def convert_to_tfrecords(json_path, image_dir, label_dir, output_path):
    # Criar o diretório para os arquivos TFRecord, caso não exista
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directory {output_dir} created.")

    writer = tf.io.TFRecordWriter(output_path)

    # Carregar os dados do JSON
    with open(json_path, 'r') as json_file:
        data = json.load(json_file)

    # Percorrer cada item no JSON
    for image_info in data['images']:
        image_path = os.path.join(image_dir, image_info['file_name'])
        label_path = os.path.join(label_dir, os.path.basename(image_info['file_name']).replace('.jpg', '.txt'))

        if os.path.exists(image_path) and os.path.exists(label_path):
            tf_example = create_tf_example(image_info, label_path)
            writer.write(tf_example.SerializeToString())

    writer.close()

if __name__ == '__main__':
    json_path = '~/Documents/trabalho visao/dataset/Saida/labels/annotations.json'
    image_dir = '~/Documents/trabalho visao/dataset/Saida/images/train'
    label_dir = '~/Documents/trabalho visao/dataset/Saida/labels/train'
    output_path = '~/Documents/trabalho visao/dataset/Saida/tfrecords/train.tfrecord'

    convert_to_tfrecords(json_path, image_dir, label_dir, output_path)
