use rune::compile::ItemBuf;
use rune::Hash;

fn main() {
    println!("{}", Hash::type_hash(&ItemBuf::with_item(["Foo", "new"])));
    println!("{}", Hash::type_hash(["Foo", "new"]));
}
