const path = require("path");
const HtmlWebpackPlugin = require("html-webpack-plugin");

module.exports = (env = {}, argv = {}) => {
  const mode = argv.mode ?? "development";
  const isProduction = mode === "production";
  const publicPath = env.publicPath ?? "/";

  return {
    mode,
    entry: "./src/main.ts",
    devtool: isProduction ? "source-map" : "eval-cheap-module-source-map",
    output: {
      clean: true,
      filename: isProduction ? "assets/[name].[contenthash].js" : "assets/[name].js",
      path: path.resolve(__dirname, "dist"),
      publicPath
    },
    resolve: {
      extensions: [".ts", ".js"]
    },
    module: {
      rules: [
        {
          test: /\.ts$/,
          exclude: /node_modules/,
          use: "ts-loader"
        },
        {
          test: /\.css$/i,
          use: ["style-loader", "css-loader"]
        },
        {
          test: /\.wgsl$/,
          type: "asset/source"
        },
        {
          test: /\.obj$/,
          type: "asset/source"
        },
        {
          test: /\.ply$/,
          type: "asset/source"
        },
        {
          test: /\.xyzn$/,
          type: "asset/source"
        }
      ]
    },
    plugins: [
      new HtmlWebpackPlugin({
        template: "./src/index.html"
      })
    ],
    devServer: {
      static: {
        directory: path.resolve(__dirname, "dist")
      },
      host: "0.0.0.0",
      port: 8000,
      open: false,
      hot: true
    }
  };
};
