import streamlit as st
import os
import tempfile
import zipfile
from pathlib import Path

import rasterio
from rasterio.features import geometry_mask, rasterize, shapes
import geopandas as gpd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from scipy.ndimage import label as ndi_label
from scipy.ndimage import sum_labels
from shapely.geometry import shape
from shapely.ops import unary_union

st.set_page_config(page_title="Классификация воды в карьере", layout="centered", page_icon="🛰️")
st.title("🛰️ Классификация воды в карьере")
st.markdown("**Загрузите файлы → укажите параметры → получите готовый вектор**  \n"
            "Работает полностью в браузере. Ничего устанавливать не нужно.")

# ====================== ФОРМА ======================
st.subheader("📤 Загрузка файлов")

col1, col2 = st.columns(2)
with col1:
    raster_file = st.file_uploader(
        "Снимок (RGBN) — .tif или .tiff",
        type=["tif", "tiff"],
        help="4-канальный снимок"
    )
with col2:
    quarry_files = st.file_uploader(
        "Вектор карьера (.shp + все сопутствующие файлы)",
        type=["shp", "shx", "dbf", "prj", "cpg"],
        accept_multiple_files=True,
        help="Загрузите ВСЕ файлы шейпфайла (shp, shx, dbf, prj...)"
    )

training_files = st.file_uploader(
    "Эталоны (.shp + все сопутствующие файлы)",
    type=["shp", "shx", "dbf", "prj", "cpg"],
    accept_multiple_files=True,
    help="Загрузите ВСЕ файлы шейпфайла эталонов"
)

st.subheader("⚙️ Параметры")
col3, col4 = st.columns(2)
with col3:
    water_class = st.number_input("Класс воды", value=9, min_value=1, step=1)
    smoothing_distance = st.number_input("Сглаживание границ (метры)", value=3.0, step=0.5, min_value=0.0)
with col4:
    quarry_class = st.number_input("Класс карьера (без воды)", value=45, min_value=1, step=1)
    min_water_size = st.number_input("Удалять водные объекты меньше (пикселей)", value=30, min_value=1, step=1)

label_field = st.text_input("Название поля с классом в эталонах", value="class")

# ====================== ЗАПУСК ======================
if st.button("🚀 Запустить классификацию", type="primary", use_container_width=True):
    if not (raster_file and quarry_files and training_files):
        st.error("❌ Загрузите все файлы!")
        st.stop()

    with st.spinner("Обработка снимка... (обычно 30–90 секунд)"):
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)

                # 1. Сохраняем снимок
                raster_path = tmp_path / raster_file.name
                with open(raster_path, "wb") as f:
                    f.write(raster_file.getbuffer())

                # 2. Сохраняем шейпфайлы
                def save_shapefile(uploaded_list, base_folder):
                    for uploaded in uploaded_list:
                        file_path = base_folder / uploaded.name
                        with open(file_path, "wb") as f:
                            f.write(uploaded.getbuffer())

                save_shapefile(quarry_files, tmp_path)
                save_shapefile(training_files, tmp_path)

                # Пути к .shp файлам
                quarry_shp = next(tmp_path.glob("*.shp"))
                training_shp = next(tmp_path.glob("*эталоны*.shp")) if any("эталоны" in f.name.lower() for f in training_files) else next(tmp_path.glob("*.shp"), None)
                if not training_shp or training_shp == quarry_shp:
                    training_shp = [f for f in tmp_path.glob("*.shp") if f != quarry_shp][0]

                # ====================== ОСНОВНАЯ ОБРАБОТКА ======================
                with rasterio.open(raster_path) as src:
                    data = src.read()
                    transform = src.transform
                    height, width = src.height, src.width
                    meta = src.meta.copy()
                    nodata_val = src.nodata
                    crs = src.crs

                # Проверка и исправление проекции
                gdf_quarry = gpd.read_file(quarry_shp)
                gdf_train = gpd.read_file(training_shp)
                if gdf_quarry.crs != crs:
                    gdf_quarry = gdf_quarry.to_crs(crs)
                if gdf_train.crs != crs:
                    gdf_train = gdf_train.to_crs(crs)

                # Проверка пустых значений в эталонах
                if gdf_train[label_field].isna().any():
                    gdf_train = gdf_train[gdf_train[label_field].notna()].copy()

                # Маска карьера и nodata
                inside_mask = ~geometry_mask(list(gdf_quarry.geometry), out_shape=(height, width),
                                             transform=transform, all_touched=True, invert=False)

                if nodata_val is not None:
                    nodata_mask = np.any(data == nodata_val, axis=0) if not np.isnan(nodata_val) else np.any(np.isnan(data), axis=0)
                else:
                    nodata_mask = np.zeros((height, width), dtype=bool)

                # Растеризация эталонов
                train_shapes = [(geom, int(label)) for geom, label in zip(gdf_train.geometry, gdf_train[label_field])]
                label_raster = rasterize(train_shapes, out_shape=(height, width), transform=transform, fill=0, dtype=np.uint8)

                # Обучение RandomForest
                valid_train_mask = (label_raster > 0) & inside_mask & ~nodata_mask
                train_features = data[:, valid_train_mask].T
                train_labels = label_raster[valid_train_mask]

                clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight='balanced')
                clf.fit(train_features, train_labels)

                # Классификация
                classify_mask = inside_mask & ~nodata_mask
                predictions = clf.predict(data[:, classify_mask].T)

                class_raster = np.zeros((height, width), dtype=np.uint8)
                class_raster[classify_mask] = predictions

                # Объединение 1+3 → вода
                class_raster[class_raster == 3] = 1

                # Чистка мелких объектов
                water_mask = (class_raster == 1) & classify_mask
                labeled_array, num_features = ndi_label(water_mask)
                if num_features > 0:
                    sizes = sum_labels(water_mask.astype(int), labeled_array, index=np.arange(1, num_features + 1))
                    small_components = np.where(sizes < min_water_size)[0] + 1
                    if len(small_components) > 0:
                        class_raster[np.isin(labeled_array, small_components)] = 2

                # Векторизация воды
                shape_gen = shapes((class_raster == 1).astype(np.uint8), mask=(class_raster == 1), transform=transform)
                geoms = [shape(geom) for geom, val in shape_gen if val == 1]
                water_gdf = gpd.GeoDataFrame(geometry=geoms, crs=crs) if geoms else gpd.GeoDataFrame(geometry=[], crs=crs)

                # Сглаживание
                if len(water_gdf) > 0:
                    smoothed = water_gdf.geometry.buffer(smoothing_distance).buffer(-smoothing_distance)
                    smoothed = smoothed.simplify(tolerance=0.5)
                    water_gdf.geometry = smoothed
                    water_gdf = water_gdf[~water_gdf.geometry.is_empty]

                # Вырезание воды из карьера + финальный вектор
                water_union = water_gdf.geometry.unary_union if len(water_gdf) > 0 else None
                quarry_polygons = []
                for geom in gdf_quarry.geometry:
                    if water_union and not water_union.is_empty:
                        diff = geom.difference(water_union)
                        if not diff.is_empty:
                            if diff.geom_type == 'Polygon':
                                quarry_polygons.append(diff)
                            elif diff.geom_type == 'MultiPolygon':
                                quarry_polygons.extend(diff.geoms)
                    else:
                        quarry_polygons.append(geom)

                final_geoms = quarry_polygons + list(water_gdf.geometry)
                final_classes = [quarry_class] * len(quarry_polygons) + [water_class] * len(water_gdf)

                final_gdf = gpd.GeoDataFrame(geometry=final_geoms, crs=crs)
                final_gdf['class'] = final_classes

                # Сохраняем результат
                base_name = raster_file.name.rsplit('.', 1)[0]
                result_shp = tmp_path / f"{base_name}_class.shp"
                final_gdf.to_file(result_shp)

                # Создаём ZIP
                zip_path = tmp_path / f"{base_name}_class.zip"
                with zipfile.ZipFile(zip_path, 'w') as z:
                    for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
                        file = result_shp.with_suffix(ext)
                        if file.exists():
                            z.write(file, file.name)

                # Читаем ZIP для скачивания
                with open(zip_path, "rb") as f:
                    zip_bytes = f.read()

                st.success("✅ Обработка завершена!")
                st.download_button(
                    label="📥 Скачать результат (ZIP с вектором)",
                    data=zip_bytes,
                    file_name=f"{base_name}_class.zip",
                    mime="application/zip",
                    use_container_width=True
                )

                st.balloons()

        except Exception as e:
            st.error(f"❌ Ошибка: {str(e)}")
            st.info("Если ошибка повторяется — пришлите мне скриншот, поправлю за минуту.")

st.caption("Создано специально для вас. Работает на Streamlit Cloud (бесплатно).")