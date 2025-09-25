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
import json

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)


# ==================== ROUTE: Harga Dogecoin ====================
@app.route("/doge-price", methods=["GET"])
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

        data.reset_index(inplace=True)
        data = data[["Date", "Open", "High", "Low", "Close", "Volume"]]

        filepath = os.path.join(DATA_DIR, "doge_prices.csv")
        data.to_csv(filepath, index=False)

        return jsonify(
            {
                "status": "success",
                "message": f"Data harga Dogecoin disimpan ke {filepath}",
                "rows": len(data),
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/musk-tweets", methods=["GET"])
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
                jsonify(
                    {
                        "error": "musk_tweets.csv belum tersedia, jalankan /musk-tweets dulu"
                    }
                ),
                404,
            )

        df = pd.read_csv(input_path, parse_dates=["createdAt"])

        analyzer = SentimentIntensityAnalyzer()

        def get_sentiment(text):
            score = analyzer.polarity_scores(str(text))["compound"]
            if score > 0.05:
                return "positive"
            elif score < -0.05:
                return "negative"
            else:
                return "neutral"

        df["sentiment"] = df["fullText"].apply(get_sentiment)

        # Simpan dengan engagement + sentiment
        output_path = os.path.join(DATA_DIR, "musk_tweets_sentiment.csv")
        df.to_csv(output_path, index=False)

        return jsonify({"status": "success", "rows": len(df), "file": output_path})

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
        try:
            threshold = float(request.args.get("threshold", 0.01))  # default 1%
        except ValueError:
            return jsonify({"error": "threshold harus berupa angka"}), 400

        try:
            lag = int(request.args.get("lag", 0))  # default tanpa lag
        except ValueError:
            return jsonify({"error": "lag harus berupa integer"}), 400

        scale_flag = request.args.get("scale", "true").lower() == "true"

        # ===== Load harga Dogecoin =====
        df_doge = pd.read_csv(doge_path, parse_dates=["Date"])
        numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
        for col in numeric_cols:
            df_doge[col] = pd.to_numeric(df_doge[col], errors="coerce")

        df_doge["Return"] = df_doge["Close"].pct_change()

        # Label harga sesuai threshold
        def price_label(x, threshold=threshold):
            if x > threshold:
                return "up"
            elif x < -threshold:
                return "down"
            else:
                return "neutral"

        df_doge["Movement"] = df_doge["Return"].apply(price_label)

        # Jika ada lag, geser target
        if lag > 0:
            df_doge["Movement_future"] = df_doge["Movement"].shift(-lag)
        else:
            df_doge["Movement_future"] = df_doge["Movement"]

        # ===== Tambahan fitur harga =====
        df_doge["VolumeChange"] = df_doge["Volume"].pct_change().fillna(0)
        df_doge["Volatility"] = df_doge["High"] - df_doge["Low"]
        df_doge["MA_3"] = df_doge["Close"].rolling(3).mean().fillna(method="bfill")
        df_doge["Volatility_5"] = df_doge["Return"].rolling(5).std().fillna(method="bfill")

        # ===== Load sentimen tweet =====
        df_tweets = pd.read_csv(tweet_path, parse_dates=["createdAt"])
        df_tweets["date"] = df_tweets["createdAt"].dt.date

        engagement_cols = ["likeCount", "retweetCount", "replyCount", "quoteCount", "viewCount"]
        for col in engagement_cols:
            if col in df_tweets.columns:
                df_tweets[col] = pd.to_numeric(df_tweets[col], errors="coerce").fillna(0)
            else:
                df_tweets[col] = 0  # fallback jika kolom tidak ada

        sentiment_daily = df_tweets.groupby("date")["sentiment"].value_counts().unstack(fill_value=0).reset_index()
        tweet_metrics = df_tweets.groupby("date")[engagement_cols].sum().reset_index()

        # Gabungkan data harga + sentimen + metrics
        df_doge["date"] = df_doge["Date"].dt.date
        df_merge = pd.merge(df_doge, sentiment_daily, on="date", how="left").fillna(0)
        df_merge = pd.merge(df_merge, tweet_metrics, on="date", how="left").fillna(0)

        # Pastikan numerik
        num_cols = [
            "positive", "negative", "neutral",
            "VolumeChange", "Volatility", "MA_3", "Volatility_5"
        ] + engagement_cols

        for col in num_cols:
            df_merge[col] = pd.to_numeric(df_merge[col], errors="coerce").fillna(0)

        # Bersihkan target
        df_merge = df_merge.dropna(subset=["Movement_future"])
        df_merge["Movement_future"] = df_merge["Movement_future"].astype(str)
        valid_labels = ["up", "down", "neutral"]
        df_merge = df_merge[df_merge["Movement_future"].isin(valid_labels)]

        # Distribusi kelas sebelum SMOTE
        class_dist_before = df_merge["Movement_future"].value_counts().to_dict()

        # ===== Fitur & Target =====
        features = num_cols
        X = df_merge[features]
        y = df_merge["Movement_future"]
        dates = df_merge["date"]

        # ===== Oversampling dengan SMOTE (hanya X, y) =====
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        import numpy as np, collections
        class_dist_after = dict(collections.Counter(y_resampled))

        # ===== Sinkronkan date dengan hasil SMOTE =====
        dates_resampled = np.random.choice(dates, size=len(y_resampled), replace=True)

        # ===== Scaling jika diaktifkan =====
        if scale_flag:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_resampled = scaler.fit_transform(X_resampled)


        X_train, X_test, y_train, y_test, date_train, date_test = train_test_split(
            X_resampled, y_resampled, dates_resampled,
            test_size=0.2, shuffle=True, random_state=42
        )

        # ===== Train Logistic Regression =====
        model = LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            max_iter=1000,
            class_weight="balanced"
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        # ===== Simpan hasil prediksi dengan tanggal =====
        result_df = pd.DataFrame(X_test, columns=features)
        result_df["date"] = pd.to_datetime(date_test).strftime("%Y-%m-%d")
        result_df["actual"] = y_test
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

        json_path = os.path.join(DATA_DIR, "logistic_regression_results.json")
        with open(json_path, "w") as f:
            json.dump(result_json, f, indent=4, default=str)
        
        return jsonify(result_json)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/logistic-regression-results', methods=['GET'])
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

        # Load data tweet
        df_tweets = pd.read_csv(tweet_path, parse_dates=["createdAt"])
        total_rows = len(df_tweets)
        total_pages = (total_rows + limit - 1) // limit  # ceil division

        # Ambil subset sesuai pagination
        df_page = df_tweets.iloc[offset:offset + limit].copy()

        # Pastikan tidak ada NaN di subset
        df_page = df_page.where(pd.notnull(df_page), None)

        # Hitung distribusi sentimen
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
            "sentiment_distribution": sentiment_counts,
            "sentiment_percentage": sentiment_percentage,
            "data": data
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route('/check-missing-tweet-cols', methods=['GET'])
def check_missing_tweet_cols():
    try:
        tweet_path = os.path.join(DATA_DIR, "musk_tweets_sentiment.csv")

        if not os.path.exists(tweet_path):
            return jsonify({"error": "musk_tweets_sentiment.csv belum tersedia, jalankan /analyze-sentiment dulu"}), 404

        # Load data
        df_tweets = pd.read_csv(tweet_path, parse_dates=["createdAt"])

        # Hitung jumlah missing values per kolom
        missing_counts = df_tweets.isna().sum().to_dict()

        # Ambil hanya kolom yang ada NaN
        missing_filtered = {k: v for k, v in missing_counts.items() if v > 0}

        return jsonify({
            "status": "success",
            "total_rows": len(df_tweets),
            "missing_columns": missing_filtered
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    app.run(debug=True, port=5000)
