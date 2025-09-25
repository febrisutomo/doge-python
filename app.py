from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import StandardScaler
import json
from imblearn.over_sampling import SMOTE
import collections
import numpy as np


app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)


@app.route("/scrap-doge-prices", methods=["GET"])
def get_doge_price():
    try:
        start_date = request.args.get("start")
        end_date = request.args.get("end")
        interval = request.args.get("interval", "1d")

        if not start_date:
            return jsonify({"error": "Parameter 'start' harus disediakan"}), 400

        if not end_date or end_date.lower() == "today":
            end_date = datetime.today().strftime("%Y-%m-%d")

        ticker = "DOGE-USD"
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval)

        if data.empty:
            return jsonify({"error": "Data tidak ditemukan"}), 404

        # Reset index supaya Date jadi kolom
        data.reset_index(inplace=True)

        # Flatten MultiIndex kalau ada
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [c[0] if isinstance(c, tuple) else c for c in data.columns]

        # Ambil hanya kolom yang kita butuhkan
        keep_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
        df = data[keep_cols].copy()

        # Pastikan numeric
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # Simpan ke CSV
        filepath = os.path.join(DATA_DIR, "doge_prices.csv")
        df.to_csv(filepath, index=False)

        return jsonify({
            "status": "success",
            "message": f"Data harga Dogecoin disimpan ke {filepath}",
            "rows": len(df),
            "columns": df.columns.tolist(),
            "preview": df.head(3).to_dict(orient="records")  # preview 3 baris
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500




@app.route("/get-musk-tweets", methods=["GET"])
def get_musk_tweets():
    try:
        start_date = request.args.get("start")
        end_date = request.args.get("end")

        if not start_date:
            return jsonify({"error": "Parameter 'start' harus disediakan"}), 400

        if not end_date or end_date.lower() == "today":
            end_date = datetime.today().strftime("%Y-%m-%d")

        csv_path = os.path.join(BASE_DIR, "all_musk_posts.csv")
        if not os.path.exists(csv_path):
            return jsonify({"error": "File all_musk_posts.csv tidak ditemukan"}), 404

        df = pd.read_csv(csv_path)
        df["createdAt"] = pd.to_datetime(df["createdAt"], errors="coerce")

        mask = (df["createdAt"] >= start_date) & (df["createdAt"] <= end_date)
        df = df.loc[mask]

        relevant_cols = [
            "id",
            "createdAt",
            "fullText",
            "likeCount",
            "retweetCount",
            "replyCount",
            "quoteCount",
            "viewCount",
            "isReply",
            "isRetweet",
            "isQuote",
        ]
        df = df[relevant_cols]
        df = df[(df["isRetweet"] != True) & (df["isReply"] != True)]
        df = df.sort_values(by="createdAt", ascending=True)

        if df.empty:
            return (
                jsonify({"error": "Tidak ada tweet pada rentang tanggal tersebut"}),
                404,
            )

        filepath = os.path.join(DATA_DIR, "musk_tweets.csv")
        df.to_csv(filepath, index=False)

        return jsonify(
            {
                "status": "success",
                "message": f"Data tweet disimpan ke {filepath}",
                "rows": len(df),
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/analyze-sentiment", methods=["GET"])
def analyze_sentiment():
    try:
        input_path = os.path.join(DATA_DIR, "musk_tweets.csv")
        if not os.path.exists(input_path):
            return (
                jsonify({"error": "musk_tweets.csv belum tersedia, jalankan /musk-tweets dulu"}),
                404,
            )

        # Load hanya kolom fullText + createdAt (opsional untuk timeline)
        df = pd.read_csv(input_path, usecols=["id", "createdAt", "fullText"])

        analyzer = SentimentIntensityAnalyzer()

        def get_sentiment(text):
            score = analyzer.polarity_scores(str(text))["compound"]
            if score > 0.05:
                return "positive"
            elif score < -0.05:
                return "negative"
            else:
                return "neutral"

        # Analisis hanya dari fullText
        df["sentiment"] = df["fullText"].apply(get_sentiment)

        # Simpan hasil (id, createdAt, fullText, sentiment)
        output_path = os.path.join(DATA_DIR, "musk_tweets_sentiment.csv")
        df.to_csv(output_path, index=False)

        return jsonify({
            "status": "success",
            "rows": len(df),
            "file": output_path,
            "columns": df.columns.tolist()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route('/analyze-logistic-regression', methods=['GET'])
def analyze_logistic_regression():
    try:
        doge_path = os.path.join(DATA_DIR, "doge_prices.csv")
        tweet_path = os.path.join(DATA_DIR, "musk_tweets_sentiment.csv")

        if not os.path.exists(doge_path):
            return jsonify({"error": "doge_prices.csv belum tersedia, jalankan /doge-price dulu"}), 404
        if not os.path.exists(tweet_path):
            return jsonify({"error": "musk_tweets_sentiment.csv belum tersedia, jalankan /analyze-sentiment dulu"}), 404

        # ===== Ambil parameter threshold, lag, scale =====
        threshold = float(request.args.get("threshold", 0.01))  # default 1%
        lag = int(request.args.get("lag", 0))  # default tanpa lag
        scale_flag = request.args.get("scale", "true").lower() == "true"

        # ===== Load harga Dogecoin =====
        df_doge = pd.read_csv(doge_path, parse_dates=["Date"])
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df_doge[col] = pd.to_numeric(df_doge[col], errors="coerce")

        df_doge["Return"] = df_doge["Close"].pct_change()

        def price_label(x, threshold=threshold):
            if x > threshold:
                return "up"
            elif x < -threshold:
                return "down"
            else:
                return "neutral"

        df_doge["Movement"] = df_doge["Return"].apply(price_label)

        if lag > 0:
            df_doge["Movement_future"] = df_doge["Movement"].shift(-lag)
        else:
            df_doge["Movement_future"] = df_doge["Movement"]

        # ===== Tambahan fitur harga =====
        df_doge["VolumeChange"] = df_doge["Volume"].pct_change().fillna(0)
        df_doge["Volatility"] = df_doge["High"] - df_doge["Low"]
        df_doge["MA_3"] = df_doge["Close"].rolling(3).mean().fillna(method="bfill")
        df_doge["Volatility_5"] = df_doge["Return"].rolling(5).std().fillna(method="bfill")

        # ===== Load sentiment tweet =====
        df_tweets = pd.read_csv(tweet_path, parse_dates=["createdAt"])
        df_tweets["date"] = df_tweets["createdAt"].dt.date

        sentiment_daily = (
            df_tweets.groupby("date")["sentiment"]
            .value_counts()
            .unstack(fill_value=0)
            .reset_index()
        )

        df_doge["date"] = df_doge["Date"].dt.date
        df_merge = pd.merge(df_doge, sentiment_daily, on="date", how="left").fillna(0)

        num_cols = ["positive", "negative", "neutral",
                    "VolumeChange", "Volatility", "MA_3", "Volatility_5"]
        for col in num_cols:
            df_merge[col] = pd.to_numeric(df_merge[col], errors="coerce").fillna(0)

        df_merge = df_merge.dropna(subset=["Movement_future"])
        df_merge["Movement_future"] = df_merge["Movement_future"].astype(str)
        valid_labels = ["up", "down", "neutral"]
        df_merge = df_merge[df_merge["Movement_future"].isin(valid_labels)]

        class_dist_before = df_merge["Movement_future"].value_counts().to_dict()

        # ===== Features & Target =====
        features = num_cols
        X = df_merge[features]
        y = df_merge["Movement_future"]
        dates = df_merge["date"]

        # ===== SMOTE =====
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        class_dist_after = dict(collections.Counter(y_resampled))

        # Simpan dates dengan index agar tetap sinkron
        dates_resampled = pd.Series(
            np.random.choice(dates, size=len(y_resampled), replace=True),
            index=y_resampled.index
        )

        # ===== Scaling =====
        if scale_flag:
            scaler = StandardScaler()
            X_resampled = scaler.fit_transform(X_resampled)

        # ===== Train-test split =====
        X_train, X_test, y_train, y_test, date_train, date_test = train_test_split(
            X_resampled, y_resampled, dates_resampled,
            test_size=0.2, shuffle=True, random_state=42
        )

        # ===== Model =====
        model = LogisticRegression(
            multi_class="multinomial", solver="lbfgs",
            max_iter=1000, class_weight="balanced"
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        # ===== Save results with aligned index =====
        result_df = pd.DataFrame(X_test, columns=features, index=y_test.index)
        result_df["date"] = pd.to_datetime(date_test.values).strftime("%Y-%m-%d")
        result_df["actual"] = y_test.values
        result_df["predicted"] = y_pred
        result_path = os.path.join(DATA_DIR, "logistic_regression_results.csv")
        result_df.to_csv(result_path, index=False)

        result_json = {
            "status": "success",
            "model": "Multinomial Logistic Regression",
            "threshold": threshold,
            "lag": lag,
            "accuracy": acc,
            "report": report,
            "rows": len(df_merge),
            "features_used": features,
            "class_distribution_before": class_dist_before,
            "class_distribution_after": class_dist_after,
            "result_file": result_path
        }

        json_path = os.path.join(DATA_DIR, "logistic_regression_evaluation.json")
        with open(json_path, "w") as f:
            json.dump(result_json, f, indent=4, default=str)

        return jsonify(result_json)

    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route('/model-evaluation', methods=['GET'])
def logistic_regression_results():
    try:
        json_path = os.path.join(DATA_DIR, "logistic_regression_results.json")

        if not os.path.exists(json_path):
            return jsonify({"error": "Belum ada hasil logistic regression. Jalankan /analyze-logistic-regression dulu."}), 404

        with open(json_path, "r") as f:
            result_json = json.load(f)

        return jsonify(result_json)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    

@app.route('/sentiment-price-line', methods=['GET'])
def sentiment_price_line():
    try:
        doge_path = os.path.join(DATA_DIR, "doge_prices.csv")
        tweet_path = os.path.join(DATA_DIR, "musk_tweets_sentiment.csv")

        if not os.path.exists(doge_path):
            return jsonify({"error": "doge_prices.csv belum tersedia, jalankan /doge-price dulu"}), 404
        if not os.path.exists(tweet_path):
            return jsonify({"error": "musk_tweets_sentiment.csv belum tersedia, jalankan /analyze-sentiment dulu"}), 404

        # Ambil parameter range (default 1D)
        range_param = request.args.get("range", "1D").upper()

        # Load harga Doge
        df_doge = pd.read_csv(doge_path, parse_dates=["Date"])
        df_doge["date"] = df_doge["Date"]

        # Load tweets + sentiment
        df_tweets = pd.read_csv(tweet_path, parse_dates=["createdAt"])
        df_tweets["date"] = df_tweets["createdAt"]

        # Hitung jumlah sentiment harian
        sentiment_daily = df_tweets.groupby([df_tweets["date"].dt.date, "sentiment"]).size().unstack(fill_value=0).reset_index()
        sentiment_daily.rename(columns={"date": "date"}, inplace=True)

        # Gabungkan harga + sentiment (daily)
        df_doge["date"] = df_doge["date"].dt.date
        df_merge = pd.merge(df_doge[["date", "Close"]], sentiment_daily, on="date", how="left").fillna(0)

        # === Resample sesuai range ===
        df_merge["date"] = pd.to_datetime(df_merge["date"])

        if range_param == "1W":
            df_grouped = df_merge.resample("W-MON", on="date").agg({
                "Close": "last",
                "positive": "sum",
                "negative": "sum",
                "neutral": "sum"
            }).reset_index()
        elif range_param == "1M":
            df_grouped = df_merge.resample("M", on="date").agg({
                "Close": "last",
                "positive": "sum",
                "negative": "sum",
                "neutral": "sum"
            }).reset_index()
        elif range_param == "1Y":
            df_grouped = df_merge.resample("Y", on="date").agg({
                "Close": "last",
                "positive": "sum",
                "negative": "sum",
                "neutral": "sum"
            }).reset_index()
        else:  # default 1D
            df_grouped = df_merge

        # Format date biar rapi
        df_grouped["date"] = df_grouped["date"].dt.strftime("%Y-%m-%d")

        # Return JSON data
        data = df_grouped.to_dict(orient="records")

        return jsonify({
            "status": "success",
            "range": range_param,
            "rows": len(df_grouped),
            "data": data
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/tweets-sentiment', methods=['GET'])
def tweets_sentiment():
    try:
        tweet_path = os.path.join(DATA_DIR, "musk_tweets_sentiment.csv")

        if not os.path.exists(tweet_path):
            return jsonify({"error": "musk_tweets_sentiment.csv belum tersedia, jalankan /analyze-sentiment dulu"}), 404

        # Ambil parameter pagination
        try:
            page = int(request.args.get("page", 1))
            limit = int(request.args.get("limit", 20))
        except ValueError:
            return jsonify({"error": "page dan limit harus berupa integer"}), 400

        if page < 1 or limit < 1:
            return jsonify({"error": "page dan limit harus lebih besar dari 0"}), 400

        offset = (page - 1) * limit

        # Ambil parameter search
        search_query = request.args.get("q", "").strip().lower()

        # Load data tweet
        df_tweets = pd.read_csv(tweet_path, parse_dates=["createdAt"])

        # Jika ada query search, filter berdasarkan fullText
        if search_query:
            df_tweets = df_tweets[df_tweets["fullText"].str.lower().str.contains(search_query, na=False)]

        total_rows = len(df_tweets)
        total_pages = (total_rows + limit - 1) // limit  # ceil division

        # Ambil subset sesuai pagination
        df_page = df_tweets.iloc[offset:offset + limit].copy()

        # Pastikan tidak ada NaN di subset
        df_page = df_page.where(pd.notnull(df_page), None)

        # Hitung distribusi sentimen (setelah filter)
        sentiment_counts = df_tweets["sentiment"].value_counts().to_dict()
        total_sentiment = sum(sentiment_counts.values())

        sentiment_percentage = {
            k: round((v / total_sentiment) * 100, 2) for k, v in sentiment_counts.items()
        }

        # Konversi ke JSON aman dari NaN
        data = df_page.to_dict(orient="records")

        return jsonify({
            "status": "success",
            "page": page,
            "limit": limit,
            "total_rows": total_rows,
            "total_pages": total_pages,
            "search_query": search_query,
            "sentiment_distribution": sentiment_counts,
            "sentiment_percentage": sentiment_percentage,
            "data": data
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route('/doge-prices', methods=['GET'])
def doge_prices():
    try:
        doge_path = os.path.join(DATA_DIR, "doge_prices.csv")

        if not os.path.exists(doge_path):
            return jsonify({
                "error": "doge_prices.csv belum tersedia, jalankan /scrap-doge-price dulu"
            }), 404

        # ===== Ambil parameter pagination =====
        try:
            page = int(request.args.get("page", 1))
            limit = int(request.args.get("limit", 50))
        except ValueError:
            return jsonify({"error": "page dan limit harus berupa integer"}), 400

        if page < 1 or limit < 1:
            return jsonify({"error": "page dan limit harus lebih besar dari 0"}), 400

        offset = (page - 1) * limit

        # ===== Ambil parameter threshold =====
        try:
            threshold = float(request.args.get("threshold", 0.05))  # default 5%
        except ValueError:
            return jsonify({"error": "threshold harus berupa angka"}), 400

        # ===== Load data =====
        df = pd.read_csv(doge_path, parse_dates=["Date"])
        df = df.sort_values("Date")

        # Pastikan numeric
        numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # Hitung return, isi NaN pertama dengan 0
        df["Return"] = df["Close"].pct_change().fillna(0)

        # Tambah kolom movement
        def price_label(x):
            if x > threshold:
                return "up"
            elif x < -threshold:
                return "down"
            else:
                return "stable"

        df["Movement"] = df["Return"].apply(price_label)

        # ===== Pagination =====
        total_rows = len(df)
        total_pages = (total_rows + limit - 1) // limit
        df_page = df.iloc[offset:offset + limit].copy()

        # Konversi ke JSON
        data = df_page.to_dict(orient="records")

        return jsonify({
            "status": "success",
            "page": page,
            "limit": limit,
            "total_rows": total_rows,
            "total_pages": total_pages,
            "threshold": threshold,
            "data": data
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/prediction-results', methods=['GET'])
def logistic_results():
    try:
        result_path = os.path.join(DATA_DIR, "logistic_regression_results.csv")

        if not os.path.exists(result_path):
            return jsonify({
                "error": "logistic_regression_results.csv belum tersedia, jalankan /analyze-logistic-regression dulu"
            }), 404

        # Ambil parameter pagination
        try:
            page = int(request.args.get("page", 1))
            limit = int(request.args.get("limit", 50))
        except ValueError:
            return jsonify({"error": "page dan limit harus berupa integer"}), 400

        if page < 1 or limit < 1:
            return jsonify({"error": "page dan limit harus lebih besar dari 0"}), 400

        offset = (page - 1) * limit

        # Load data
        df = pd.read_csv(result_path, parse_dates=["date"])

        # Pastikan kolom ada
        expected_cols = [
            "positive", "negative", "neutral",
            "VolumeChange", "Volatility", "MA_3", "Volatility_5",
            "date", "actual", "predicted"
        ]
        missing_cols = [c for c in expected_cols if c not in df.columns]
        if missing_cols:
            return jsonify({"error": f"Kolom hilang di file: {missing_cols}"}), 400

        # Urutkan berdasarkan tanggal
        df = df.sort_values("date")

        # Hitung total & pagination
        total_rows = len(df)
        total_pages = (total_rows + limit - 1) // limit

        df_page = df.iloc[offset:offset + limit].copy()

        # Konversi ke JSON
        data = df_page[expected_cols].to_dict(orient="records")

        return jsonify({
            "status": "success",
            "page": page,
            "limit": limit,
            "total_rows": total_rows,
            "total_pages": total_pages,
            "data": data
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
