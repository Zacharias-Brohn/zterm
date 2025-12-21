# Maintainer: Zach <zach@brohn.se>
pkgname=zterm
pkgver=0.1.0
pkgrel=1
pkgdesc="A GPU-accelerated terminal emulator for Wayland"
arch=('x86_64')
url="https://github.com/Zacharias-Brohn/zterm"
license=('MIT')
depends=(
	'fontconfig'
	'freetype2'
	'wayland'
	'libxkbcommon'
	'vulkan-icd-loader'
)
makedepends=('rust' 'cargo')
source=()

build() {
	cd "$startdir"
	cargo build --release --features production
}

package() {
	cd "$startdir"
	install -Dm755 "target/release/zterm" "$pkgdir/usr/bin/zterm"
	install -Dm644 "zterm.terminfo" "$pkgdir/usr/share/zterm/zterm.terminfo"

	# Compile and install terminfo
	tic -x -o "$pkgdir/usr/share/terminfo" zterm.terminfo
}
