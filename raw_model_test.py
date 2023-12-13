from sentence_transformers.cross_encoder import CrossEncoder

model = CrossEncoder('sentence-transformers/sentence-t5-base')
scores = model.predict([["Dokter", "Dokter Umum"],
                        ["Dokter", "Dokter Anak"],
                        ["Dokter", "Dokter Gigi"],
                        ["Dokter", "Dokter Medis"],
                        ["Dokter", "Dokter Gizi"]])
print(scores)
