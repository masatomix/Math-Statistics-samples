# Math-Statistics-samples
Math-Statistics-samples

https://github.com/ghmagazine/python_stat_sample

をやってます。


## Mac の環境
mojaveでした

```
$ sw_vers
ProductName:	Mac OS X
ProductVersion:	10.14.1
BuildVersion:	18B75
```

## Homebrew のインストール
https://brew.sh/index_ja を参考に、Homebrew をインストール

2018/12/02現在:

```
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
```

でよさそう。環境によって(mojaveかな？)はエラーがおこるが、

```
cd /usr/local && sudo chown -R $(whoami) bin
```

ってownerを変えればたぶん問題なし。


## Pyenv のインストール
Pythonのバージョン切替のためのツールを入れる

https://github.com/pyenv/pyenv を参考に。

2018/12/02現在:

```
$ brew install pyenv

$ echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bash_profile
$ echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bash_profile
$ echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n  eval "$(pyenv init -)"\nfi' >> ~/.bash_profile

```

でよさそう。

## Pythonのインストール

```
$ pyenv install 3.7.1

なんかエラーが出たら、たいがい
$ xcode-select --install
これもしくは、
$ sudo installer -pkg /Library/Developer/CommandLineTools/Packages/macOS_SDK_headers_for_macOS_10.14.pkg -target /
などで解決しましょう。(sudo installer もmojaveで起こるっぽい)
```


## 仮想環境を作成し、それをactivate

```
$ python3 -m venv ~/venv/venv_samples
$ source ~/venv/venv_samples/bin/activate
(venv_samples) $
```

## matplotlib のライブラリのインストールと、環境設定

```
(venv_samples) $ pip install numpy
(venv_samples) $ pip install matplotlib
(venv_samples) $ python3 sample.py   下記の .matploglib ディレクトリを作るため呼んでみる(多分エラーになる)
(venv_samples) $ echo "backend : Tkagg" >> ~/.matplotlib/matplotlibrc
```

## サンプル実行

```
(venv_samples) $ python3 sample.py
```

これでグラフが表示されればOK。


もしくは、

```
(venv_samples) $ pip install PyQt5
(venv_samples) $ echo "backend : Qt4Agg" >> ~/.matplotlib/matplotlibrc
```

とやるのでもよいっぽい。


## 日本語表示
http://bit.ly/2FQWYwa をみながら、日本語フォントをインストール後、


```
(venv_samples) $ echo "font.family: IPAexGothic" >> ~/.matplotlib/matplotlibrc
(venv_samples) $ python3 sample.py
```

これで日本語表示OK！
