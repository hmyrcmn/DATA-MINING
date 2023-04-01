import pandas as pd
import numpy as np

# Verileri oku
data = pd.read_csv('/content/DATA.csv')

# Gini Endeksi Hesaplama Fonksiyonu
def gini_index(groups, classes):
    # Tüm örneklerin sayısını bul
    n_instances = float(sum([len(group) for group in groups]))
    # Her grubun Gini skorunu başlat
    gini = 0.0
    # Her gruptaki örneklerin sayısına göre Gini skoru hesapla
    for group in groups:
        size = float(len(group))
        # Sıfıra bölme hatasını önlemek için kontrol et
        if size == 0:
            continue
        score = 0.0
        # Her sınıf için bir skor hesapla
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
        # Grup ağırlığına göre Gini skorunu hesapla
        gini += (1.0 - score) * (size / n_instances)
    return gini

# Entropi Hesaplama Fonksiyonu
def entropy(groups, classes):
    # Tüm örneklerin sayısını bul
    n_instances = float(sum([len(group) for group in groups]))
    # Her grubun entropisini başlat
    ent = 0.0
    # Her gruptaki örneklerin sayısına göre entropi hesapla
    for group in groups:
        size = float(len(group))
        # Sıfıra bölme hatasını önlemek için kontrol et
        if size == 0:
            continue
        score = 0.0
        # Her sınıf için bir skor hesapla
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            if p > 0:
                score += -p * np.log2(p)
        # Grup ağırlığına göre entropi hesapla
        ent += score * (size / n_instances)
    return ent

# (a) Eğitim örneklerinin genel koleksiyonu için Gini endeksini ve entropiyi hesaplayın.
classes = np.unique(data.iloc[:, -1])
groups = [data.values.tolist()]
gini = gini_index(groups, classes)
ent = entropy(groups, classes)
print("Genel Gini endeksi: %.3f" % gini)
print("Genel Entropi: %.3f" % ent)

# (b) Customer ID özniteliği için Gini endeksini ve entropiyi hesaplayın.
groups = []
for value in np.unique(data.iloc[:, 0]):
    groups.append(data[data.iloc[:, 0] == value].values.tolist())
gini = gini_index(groups, classes)
ent = entropy(groups, classes)
print("Customer ID Gini endeksi: %.3f" % gini)
print("Customer ID Entropi: %.3f" % ent)

#(c) Gender özelliği için Gini endeksini ve entropiyi hesaplayın.
groups = []
for value in np.unique(data.iloc[:, 1]):
  groups.append(data[data.iloc[:, 1] == value].values.tolist())
gini = gini_index(groups, classes)
ent = entropy(groups, classes)
print("Gender Gini endeksi: %.3f" % gini)
print("Gender Entropi: %.3f" % ent)

#(d) Çok yollu bölme (multiway split) kullanarak Car Type özniteliği için Gini endeksini ve entropiyi hesaplayın. groups = []
for value in np.unique(data.iloc[:, 2]):
  groups.append(data[data.iloc[:, 2] == value].values.tolist())
gini = gini_index(groups, classes)
ent = entropy(groups, classes)
print("Car Type Gini endeksi: %.3f" % gini)
print("Car Type Entropi: %.3f" % ent)

#(e) Çok yollu bölmeyi kullanarak Shirt Size özelliği için Gini endeksini ve entropiyi hesaplayın.
groups = []
for value in np.unique(data.iloc[:, 3]):
    groups.append(data[data.iloc[:, 3] == value].values.tolist())
gini = gini_index(groups, classes)
ent = entropy(groups, classes)
print("Shirt Size Gini endeksi: %.3f" % gini)
print("Shirt Size Entropi: %.3f" % ent)

#(f) Gender, Car Type veya Shirt Size hangi özellik daha iyidir?En düşük Gini endeksine sahip olan özellik en iyisidir.
gini_vals = []
gini_vals.append(gini_index(groups, classes) for groups in [groups_1, groups_2, groups_3])
min_index = np.argmin(gini_vals)
if min_index == 0:
  print("Gender özelliği en iyidir.")
elif min_index == 1:
  print("Car Type özelliği en iyidir.")
else:
  print("Shirt Size özelliği en iyidir.")