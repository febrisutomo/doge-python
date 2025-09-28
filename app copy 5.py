from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import os
from datetime import datetime
from sklearn.metrics import classification_report, accuracy_score, matthews_corrcoef, confusion_matrix, balanced_accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import StandardScaler
import json
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import TimeSeriesSplit


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
                jsonify({"error": "musk_tweets.csv belum tersedia, jalankan /get-musk-tweets dulu"}),
                404,
            )

        # Baca semua kolom (biar fleksibel)
        df = pd.read_csv(input_path)

        # Pastikan kolom wajib ada
        required_cols = ["id", "createdAt", "fullText"]
        for col in required_cols:
            if col not in df.columns:
                return jsonify({"error": f"Kolom wajib '{col}' tidak ditemukan di musk_tweets.csv"}), 400

        # Handle NaN di teks
        df["fullText"] = df["fullText"].fillna("")

        # --- Cleaning teks ---
        import re
        def clean_text(text):
            text = re.sub(r"http\S+", "", str(text))   # hapus URL
            text = re.sub(r"@\w+", "", text)           # hapus mention
            return text.strip().lower()

        df["cleanText"] = df["fullText"].apply(clean_text)

        # --- Analisis VADER ---
        analyzer = SentimentIntensityAnalyzer()
        scores = df["cleanText"].apply(analyzer.polarity_scores)

        # Ekstrak skor
        df["neg"] = scores.apply(lambda x: x["neg"])
        df["neu"] = scores.apply(lambda x: x["neu"])
        df["pos"] = scores.apply(lambda x: x["pos"])
        df["compound"] = scores.apply(lambda x: x["compound"])

        # Label sentimen sederhana
        def get_sentiment(score):
            if score > 0.05:
                return "positive"
            elif score < -0.05:
                return "negative"
            else:
                return "neutral"

        df["sentiment"] = df["compound"].apply(get_sentiment)

        # Simpan hasil ke CSV
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

        # ===== Parameter =====
        threshold = float(request.args.get("threshold", 0.02))
        scale_flag = request.args.get("scale", "true").lower() == "true"
        n_splits = int(request.args.get("cv", 5))  # default 5 folds

        # ===== Load harga Dogecoin =====
        df = pd.read_csv(doge_path, parse_dates=["Date"])
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["Return"] = df["Close"].pct_change().fillna(0)

        def price_label(x, threshold=threshold):
            if x > threshold: return "up"
            elif x < -threshold: return "down"
            return "neutral"
        df["Movement_future"] = df["Return"].apply(price_label)

        # ===== Fitur teknikal =====
        df["VolumeChange"] = df["Volume"].pct_change().fillna(0)
        df["Volatility"] = df["High"] - df["Low"]
        df["MA_3"] = df["Close"].rolling(3).mean().fillna(method="bfill")
        df["MA_7"] = df["Close"].rolling(7).mean().fillna(method="bfill")
        df["MA_14"] = df["Close"].rolling(14).mean().fillna(method="bfill")

        # MA deviation
        for ma in [3, 7, 14]:
            df[f"MA_{ma}_dev"] = (df["Close"]/df[f"MA_{ma}"] - 1).replace([np.inf, -np.inf], 0).fillna(0)

        df["Volatility_5"] = df["Return"].rolling(5).std().fillna(method="bfill")

        # RSI 14
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        df["RSI_14"] = 100 - (100/(1+rs))

        # Bollinger z-score
        rolling_mean = df["Close"].rolling(20).mean()
        rolling_std = df["Close"].rolling(20).std().replace(0, np.nan)
        df["BB_z"] = ((df["Close"] - rolling_mean) / rolling_std).fillna(0).clip(-5, 5)

        # Lagged returns
        for lag in [1, 2, 3, 5]:
            df[f"Return_lag{lag}"] = df["Return"].shift(lag).fillna(0)

        # ===== Load sentiment =====
        tweets = pd.read_csv(tweet_path, parse_dates=["createdAt"])
        tweets["date"] = tweets["createdAt"].dt.date
        # weight = tweets.get("likeCount", 1) + tweets.get("retweetCount", 1) + 1
        

        # for col in ["compound", "pos", "neg", "neu"]:
        #     tweets[f"{col}_w"] = tweets[col] * weight
        
        # Hitung bobot tweet: popularitas + sebaran + diskusi + framing
        tweets["weight"] = (
            tweets.get("likeCount", 0) +
            2 * tweets.get("retweetCount", 0) +
            tweets.get("replyCount", 0) +
            tweets.get("quoteCount", 0) +
            1
        )

        # # Kalikan skor sentimen dengan bobot
        for col in ["compound", "pos", "neg", "neu"]:
            tweets[f"{col}_w"] = tweets[col] * tweets["weight"]


        # sent = tweets.groupby("date")[
        #     ["compound_w", "pos_w", "neg_w", "neu_w", "likeCount", "retweetCount"]
        # ].sum().reset_index()

        # for col in ["compound", "pos", "neg", "neu"]:
        #     sent[col] = (
        #         sent[f"{col}_w"] / (sent["likeCount"] + sent["retweetCount"] + 1)
        #     ).fillna(0)
        sent = tweets.groupby("date")[
            ["compound_w", "pos_w", "neg_w", "neu_w", "weight"]
        ].sum().reset_index()

        for col in ["compound", "pos", "neg", "neu"]:
            sent[col] = (sent[f"{col}_w"] / sent["weight"]).fillna(0)

            sent[f"{col}_roll3"] = sent[col].rolling(3).mean().fillna(method="bfill")
            sent[f"{col}_ewm3"]  = sent[col].ewm(span=3).mean()
            sent[f"{col}_diff"]  = sent[col].diff().fillna(0)
            sent[f"{col}_surge"] = ((sent[col] - sent[col].rolling(7).mean()) /
                                    (sent[col].rolling(7).std()+1e-9))\
                                    .replace([np.inf, -np.inf], 0).fillna(0).clip(-5, 5)

        # ===== Merge =====
        df["date"] = df["Date"].dt.date
        dfm = pd.merge(df, sent, on="date", how="left").fillna(0)
        valid_labels = ["up", "down", "neutral"]
        dfm = dfm[dfm["Movement_future"].isin(valid_labels)]

        # ===== Features =====
        features = [
            "compound_roll3", "compound_ewm3", "compound_diff", "compound_surge",
            "pos_roll3", "pos_ewm3", "pos_diff",
            "neg_roll3", "neg_ewm3", "neg_diff",
            "neu_roll3", "neu_ewm3", "neu_diff",
            "VolumeChange", "Volatility", "Volatility_5",
            "MA_3_dev", "MA_7_dev", "MA_14_dev", "RSI_14", "BB_z",
            "Return_lag1", "Return_lag2", "Return_lag3", "Return_lag5"
        ]
        X = dfm[features]
        y = dfm["Movement_future"]

        # ===== VarianceThreshold =====
        vt = VarianceThreshold(1e-6)
        X = vt.fit_transform(X)
        kept_features = [f for f, keep in zip(features, vt.get_support()) if keep]

        # ===== Cross-validation =====
        tscv = TimeSeriesSplit(n_splits=n_splits)
        accs, mccs, bal_accs, macro_f1s = [], [], [], []
        fold_results = []
        coef_sum = np.zeros(len(kept_features))

        for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            if scale_flag:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

            model = LogisticRegression(
                multi_class="multinomial", solver="lbfgs",
                max_iter=1000, class_weight="balanced", C=1.0
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            mcc = matthews_corrcoef(y_test, y_pred)
            bal_acc = balanced_accuracy_score(y_test, y_pred)
            macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            cm = confusion_matrix(y_test, y_pred, labels=["up", "down", "neutral"]).tolist()

            accs.append(acc)
            mccs.append(mcc)
            bal_accs.append(bal_acc)
            macro_f1s.append(macro_f1)

            fold_results.append({
                "fold": i+1,
                "accuracy": acc,
                "mcc": mcc,
                "balanced_accuracy": bal_acc,
                "macro_f1": macro_f1,
                "report": report,
                "confusion_matrix": cm
            })
            
            coef_sum += model.coef_.mean(axis=0)

        feature_importance = dict(zip(kept_features, coef_sum/len(accs)))
        
        feature_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

        result_json = {
            "status": "success",
            "model": "Multinomial Logistic Regression",
            "threshold": threshold,
            "rows": len(dfm),
            "cv_accuracy_mean": float(np.mean(accs)),
            "cv_accuracy_std": float(np.std(accs)),
            "cv_mcc_mean": float(np.mean(mccs)),
            "cv_balanced_accuracy_mean": float(np.mean(bal_accs)),
            "cv_macro_f1_mean": float(np.mean(macro_f1s)),
            "last_fold_accuracy": float(accs[-1]),
            "last_fold_mcc": float(mccs[-1]),
            "report_last_fold": fold_results[-1]["report"],
            "confusion_matrix_last_fold": fold_results[-1]["confusion_matrix"],
            "feature_importance": feature_importance,
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
