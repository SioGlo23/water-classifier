import streamlit as st
import tempfile
from pathlib import Path
import zipfile

import rasterio
from rasterio.features import geometry_mask, rasterize, shapes
import geopandas as gpd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from scipy.ndimage import label as ndi_label
from scipy.ndimage import sum_labels
from shapely.geometry import shape

st.set_page_config(page_title="Классификация воды", page_icon="🛰️", layout="centered")
st.title("🛰️ Классификация воды в карьере")
st.markdown("Загрузи файлы — получи готовый вектор с водой (класс 9) и карьером (класс 45)")

# ====================== ФОРМА ======================
raster_file = st.file_uploader("📸 Снимок (.tif / .tiff)", type=["tif", "tiff"])

quarry_files = st.file_uploader(
    "🏗️ Вектор карьера (.shp + shx, dbf, prj...)",
    type=["shp","shx","dbf","prj","cpg"], accept_multiple_files=True
)

training_files = st.file_uploader(
    "📍 Эталоны (.shp + shx, dbf, prj...)",
    type=["shp","shx","dbf","prj","cpg"], accept_multiple_files=True
)

st.subheader("⚙️ Параметры")
col1, col2 = st.columns(2)
with col1:
    water_class = st.number_input("Класс воды", value=9)
    smoothing = st.number_input("Сглаживание (метры)", value=3.0, step=0.5)
with col2:
    quarry_class = st.number_input("Класс карьера", value=45)
    min_size = st.number_input("Мин. размер воды (пиксели)", value=30)

label_field = st.text_input("Имя поля класса в эталонах", value="class")

# ====================== ЗАЩИЩЁННЫЙ ЗАПУСК ======================
if st.button("🚀 Запустить классификацию", type="primary", use_container_width=True):
    if not (raster_file and quarry_files and training_files):
        st.error("Загрузи все файлы")
        st.stop()

    with st.spinner("Идёт обработка (30–90 сек)..."):
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp = Path(tmp_dir)

                # Сохраняем файлы
                (tmp / raster_file.name).write_bytes(raster_file.getbuffer())
                for f in quarry_files + training_files:
                    (tmp / f.name).write_bytes(f.getbuffer())

                # Находим два .shp файла (самый надёжный способ)
                shp_files = sorted(tmp.glob("*.shp"))
                if len(shp_files) < 2:
                    st.error(f"Найдено только {len(shp_files)} shp-файлов. Загрузи оба шейпфайла полностью.")
                    st.stop()

                quarry_shp = shp_files[0]
                training_shp = shp_files[1]

                # ====================== ОБРАБОТКА ======================
                with rasterio.open(tmp / raster_file.name) as src:
                    data = src.read()
                    transform = src.transform
                    height, width = src.height, src.width
                    nodata_val = src.nodata
                    crs = src.crs

                gdf_quarry = gpd.read_file(quarry_shp)
                gdf_train = gpd.read_file(training_shp)

                # Проверка поля класса
                if label_field not in gdf_train.columns:
                    st.error(f"Поле '{label_field}' не найдено! Доступные поля: {list(gdf_train.columns)}")
                    st.stop()

                # Приведение проекций
                if gdf_quarry.crs != crs: gdf_quarry = gdf_quarry.to_crs(crs)
                if gdf_train.crs != crs: gdf_train = gdf_train.to_crs(crs)

                gdf_train = gdf_train[gdf_train[label_field].notna()]

                # Маски
                inside_mask = ~geometry_mask(list(gdf_quarry.geometry), out_shape=(height, width),
                                             transform=transform, all_touched=True, invert=False)

                nodata_mask = np.any(data == nodata_val, axis=0) if nodata_val is not None else np.zeros((height, width), dtype=bool)

                # Классификация
                train_mask = (rasterize([(g, int(v)) for g,v in zip(gdf_train.geometry, gdf_train[label_field])],
                                        out_shape=(height, width), transform=transform, fill=0, dtype=np.uint8) > 0) & inside_mask & ~nodata_mask

                clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight='balanced')
                clf.fit(data[:, train_mask].T, rasterize([(g, int(v)) for g,v in zip(gdf_train.geometry, gdf_train[label_field])],
                                                         out_shape=(height, width), transform=transform, fill=0, dtype=np.uint8)[train_mask])

                class_raster = np.zeros((height, width), dtype=np.uint8)
                class_raster[inside_mask & ~nodata_mask] = clf.predict(data[:, inside_mask & ~nodata_mask].T)

                class_raster[class_raster == 3] = 1

                # Чистка и векторизация (как раньше)
                water_mask = (class_raster == 1) & (inside_mask & ~nodata_mask)
                labeled, n = ndi_label(water_mask)
                if n > 0:
                    sizes = sum_labels(water_mask.astype(int), labeled, index=np.arange(1, n+1))
                    class_raster[np.isin(labeled, np.where(sizes < min_size)[0] + 1)] = 2

                shape_gen = shapes((class_raster == 1).astype(np.uint8), mask=(class_raster == 1), transform=transform)
                geoms = [shape(geom) for geom, val in shape_gen if val == 1]
                water_gdf = gpd.GeoDataFrame(geometry=geoms, crs=crs) if geoms else gpd.GeoDataFrame(geometry=[], crs=crs)

                if len(water_gdf) > 0:
                    water_gdf.geometry = water_gdf.geometry.buffer(smoothing).buffer(-smoothing).simplify(0.5)
                    water_gdf = water_gdf[~water_gdf.geometry.is_empty]

                # Финальный вектор
                water_union = water_gdf.geometry.unary_union if len(water_gdf) > 0 else None
                final_gdf = gpd.GeoDataFrame(geometry=[], crs=crs)
                final_gdf['class'] = []

                for geom in gdf_quarry.geometry:
                    diff = geom.difference(water_union) if water_union else geom
                    if not diff.is_empty:
                        if diff.geom_type == 'Polygon':
                            final_gdf = gpd.GeoDataFrame(geometry=final_gdf.geometry.tolist() + [diff], crs=crs)
                            final_gdf['class'] = final_gdf['class'].tolist() + [quarry_class]
                        else:
                            for p in diff.geoms:
                                final_gdf = gpd.GeoDataFrame(geometry=final_gdf.geometry.tolist() + [p], crs=crs)
                                final_gdf['class'] = final_gdf['class'].tolist() + [quarry_class]

                for g in water_gdf.geometry:
                    final_gdf = gpd.GeoDataFrame(geometry=final_gdf.geometry.tolist() + [g], crs=crs)
                    final_gdf['class'] = final_gdf['class'].tolist() + [water_class]

                # Сохранение
                base = raster_file.name.rsplit('.', 1)[0]
                result_shp = tmp / f"{base}_class.shp"
                final_gdf.to_file(result_shp)

                zip_path = tmp / f"{base}_class.zip"
                with zipfile.ZipFile(zip_path, 'w') as z:
                    for ext in ['.shp','.shx','.dbf','.prj','.cpg']:
                        f = result_shp.with_suffix(ext)
                        if f.exists():
                            z.write(f, f.name)

                with open(zip_path, "rb") as f:
                    st.download_button("📥 Скачать результат", f.read(), f"{base}_class.zip", "application/zip", use_container_width=True)

                st.success("Готово! 🎉")
                st.balloons()

        except Exception as e:
            st.error("Произошла ошибка")
            st.exception(e)          # ←←← ПОЛНЫЙ TRACEBACK ПРЯМО В БРАУЗЕРЕ

